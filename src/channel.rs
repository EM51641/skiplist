use crate::book::Book;
use crate::order::{CancelOrder, ModifyOrder, NewOrder, Side};
use rust_decimal::Decimal;
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};

pub type OrderId = u64;
pub type RejectReason = String;
pub type CancelReason = String;

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct EngineSnapshot {
    pub bids: crate::view::BookView,
    pub asks: crate::view::BookView,
}

pub enum Command {
    NewOrder {
        order_id: OrderId,
        side: Side,
        price: Decimal,
        qty: i32,
        reply: oneshot::Sender<OrderAck>,
    },
    CancelOrder {
        order_id: OrderId,
        reply: oneshot::Sender<OrderAck>,
    },
    ModifyOrder {
        order_id: OrderId,
        price: Decimal,
        qty: i32,
        reply: oneshot::Sender<OrderAck>,
    },

    #[cfg(test)]
    Snapshot {
        reply: oneshot::Sender<EngineSnapshot>,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum OrderAck {
    Accepted {
        order_id: OrderId,
    },
    Rejected {
        reason: RejectReason,
    },
    Fill {
        order_id: OrderId,
        qty: i32,
        px: Decimal,
        remaining: i32,
    },
    Cancelled {
        order_id: OrderId,
        reason: CancelReason,
    },
    Modified {
        order_id: OrderId,
        price: Decimal,
        qty: i32,
    },
}

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<Command>,
}

impl EngineHandle {
    pub async fn submit_order(
        &self,
        side: Side,
        order_id: OrderId,
        price: Decimal,
        qty: i32,
    ) -> OrderAck {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::NewOrder {
                order_id,
                side,
                price,
                qty,
                reply,
            })
            .await
            .expect("engine dropped");

        rx.await.expect("engine dropped before replying")
    }

    pub async fn cancel_order(&self, order_id: OrderId) -> OrderAck {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::CancelOrder { order_id, reply })
            .await
            .expect("engine dropped");

        rx.await.expect("engine dropped before replying")
    }

    pub async fn modify_order(&self, order_id: OrderId, price: Decimal, qty: i32) -> OrderAck {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::ModifyOrder {
                order_id,
                price,
                qty,
                reply,
            })
            .await
            .expect("engine dropped");

        rx.await.expect("engine dropped before replying")
    }

    #[cfg(test)]
    pub async fn snapshot(&self) -> EngineSnapshot {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::Snapshot { reply })
            .await
            .expect("engine dropped");

        rx.await.expect("engine dropped before replying")
    }
}

fn pick<'a>(side: Side, bids: &'a mut Book, asks: &'a mut Book) -> &'a mut Book {
    match side {
        Side::Buy => bids,
        Side::Sell => asks,
    }
}

pub fn spawn_engine() -> EngineHandle {
    let (tx, mut rx) = mpsc::channel::<Command>(1024);

    tokio::spawn(async move {
        let mut bids = Book::new(Side::Buy, None, None, None);
        let mut asks = Book::new(Side::Sell, None, None, None);
        let mut side_of: HashMap<OrderId, Side> = HashMap::new();

        while let Some(cmd) = rx.recv().await {
            match cmd {
                Command::NewOrder {
                    order_id,
                    side,
                    price,
                    qty,
                    reply,
                } => {
                    // Cross the incoming order against the opposite book first,
                    // then rest whatever quantity is left unfilled on its own side.
                    let unfilled = match side {
                        Side::Buy => asks.book_matcher(&price, qty),
                        Side::Sell => bids.book_matcher(&price, qty),
                    };
                    if unfilled > 0 {
                        side_of.insert(order_id, side);
                        let book = pick(side, &mut bids, &mut asks);
                        book.submit_order(NewOrder {
                            id: order_id,
                            side,
                            price,
                            qty: unfilled,
                        });
                    }
                    let _ = reply.send(OrderAck::Accepted { order_id });
                }
                Command::CancelOrder { order_id, reply } => match side_of.remove(&order_id) {
                    Some(side) => {
                        let book = pick(side, &mut bids, &mut asks);
                        book.cancel_order(CancelOrder { id: order_id });
                        let _ = reply.send(OrderAck::Cancelled {
                            order_id,
                            reason: format!("{order_id} Cancelled"),
                        });
                    }
                    None => {
                        let _ = reply.send(OrderAck::Rejected {
                            reason: format!("unknown order id:{order_id}"),
                        });
                    }
                },
                Command::ModifyOrder {
                    order_id,
                    price,
                    qty,
                    reply,
                } => match side_of.get(&order_id).copied() {
                    Some(side) => {
                        let (old_price, old_remaining) = {
                            let book = pick(side, &mut bids, &mut asks);
                            book.resting_order(order_id).unwrap()
                        };
                        // A reprice or a size increase forfeits time priority, so
                        // it re-enters the market as an aggressor: pull the resting
                        // order, cross against the opposite book, rest the rest.
                        // A pure size decrease at the same price stays in place.
                        let aggressive = price != old_price || qty > old_remaining;
                        if aggressive {
                            pick(side, &mut bids, &mut asks)
                                .cancel_order(CancelOrder { id: order_id });
                            let unfilled = match side {
                                Side::Buy => asks.book_matcher(&price, qty),
                                Side::Sell => bids.book_matcher(&price, qty),
                            };
                            if unfilled > 0 {
                                pick(side, &mut bids, &mut asks).submit_order(NewOrder {
                                    id: order_id,
                                    side,
                                    price,
                                    qty: unfilled,
                                });
                            } else {
                                side_of.remove(&order_id);
                            }
                        } else {
                            pick(side, &mut bids, &mut asks).modify_order(ModifyOrder {
                                id: order_id,
                                new_price: price,
                                new_qty: qty,
                            });
                        }
                        let _ = reply.send(OrderAck::Modified {
                            order_id,
                            price,
                            qty,
                        });
                    }
                    None => {
                        let _ = reply.send(OrderAck::Rejected {
                            reason: format!("unknown order id:{order_id}"),
                        });
                    }
                },
                #[cfg(test)]
                Command::Snapshot { reply } => {
                    let snapshot = EngineSnapshot {
                        bids: crate::view::BookView::from_book(&bids),
                        asks: crate::view::BookView::from_book(&asks),
                    };
                    let _ = reply.send(snapshot);
                }
            }
        }
    });

    EngineHandle { tx }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{BookView, OrderView, PriceLevelView};

    fn px(cents: i64) -> Decimal {
        Decimal::new(cents, 2)
    }

    fn book(side: Side, levels: Vec<PriceLevelView>) -> BookView {
        BookView { side, levels }
    }

    fn level(price: Decimal, total_qty: i32, orders: Vec<OrderView>) -> PriceLevelView {
        PriceLevelView {
            price,
            total_qty,
            orders,
        }
    }

    fn order(id: OrderId, side: Side, price: Decimal, qty: i32, remaining: i32) -> OrderView {
        OrderView {
            id,
            side,
            price,
            qty,
            remaining,
        }
    }

    fn equivalent(side: Side, orders: &[(OrderId, Decimal, i32)]) -> BookView {
        let mut book = Book::new(side, None, None, None);
        for &(id, price, qty) in orders {
            book.submit_order(NewOrder {
                id,
                side,
                price,
                qty,
            });
        }
        BookView::from_book(&book)
    }

    #[tokio::test]
    async fn new_order_appears_in_snapshot() {
        let engine = spawn_engine();

        assert_eq!(
            engine.submit_order(Side::Buy, 1, px(10000), 100).await,
            OrderAck::Accepted { order_id: 1 }
        );

        assert_eq!(
            engine.submit_order(Side::Sell, 2, px(10100), 50).await,
            OrderAck::Accepted { order_id: 2 }
        );

        let snapshot = engine.snapshot().await;

        assert_eq!(
            snapshot.bids,
            book(
                Side::Buy,
                vec![level(
                    px(10000),
                    100,
                    vec![order(1, Side::Buy, px(10000), 100, 100)]
                )]
            )
        );
        assert_eq!(
            snapshot.asks,
            book(
                Side::Sell,
                vec![level(
                    px(10100),
                    50,
                    vec![order(2, Side::Sell, px(10100), 50, 50)]
                )]
            )
        );
    }

    #[tokio::test]
    async fn cancel_order_disappears_from_snapshot() {
        let engine = spawn_engine();

        engine.submit_order(Side::Buy, 1, px(10000), 100).await;

        assert_eq!(
            engine.cancel_order(1).await,
            OrderAck::Cancelled {
                order_id: 1,
                reason: format!("{} Cancelled", 1),
            }
        );

        // The cancelled order is gone: the book is empty.
        let snapshot = engine.snapshot().await;
        assert_eq!(snapshot.bids, equivalent(Side::Buy, &[]));
        assert_eq!(snapshot.asks, equivalent(Side::Sell, &[]));
    }

    #[tokio::test]
    async fn cancel_unknown_order_leaves_snapshot_unchanged() {
        let engine = spawn_engine();

        engine.submit_order(Side::Sell, 1, px(10100), 50).await;

        assert_eq!(
            engine.cancel_order(999).await,
            OrderAck::Rejected {
                reason: format!("unknown order id:{}", 999),
            }
        );

        assert_eq!(
            engine.snapshot().await.asks,
            book(
                Side::Sell,
                vec![level(
                    px(10100),
                    50,
                    vec![order(1, Side::Sell, px(10100), 50, 50)]
                )]
            )
        );
    }

    #[tokio::test]
    async fn modify_order_updates_snapshot_quantity() {
        let engine = spawn_engine();

        engine.submit_order(Side::Sell, 1, px(10100), 50).await;

        assert_eq!(
            engine.modify_order(1, px(10100), 20).await,
            OrderAck::Modified {
                order_id: 1,
                price: px(10100),
                qty: 20,
            }
        );

        assert_eq!(
            engine.snapshot().await.asks,
            book(
                Side::Sell,
                vec![level(
                    px(10100),
                    20,
                    vec![order(1, Side::Sell, px(10100), 50, 20)]
                )]
            )
        );
    }

    #[tokio::test]
    async fn modify_order_reprices_in_snapshot() {
        let engine = spawn_engine();

        engine.submit_order(Side::Sell, 1, px(10100), 50).await;

        assert_eq!(
            engine.modify_order(1, px(10200), 50).await,
            OrderAck::Modified {
                order_id: 1,
                price: px(10200),
                qty: 50,
            }
        );

        assert_eq!(
            engine.snapshot().await.asks,
            book(
                Side::Sell,
                vec![level(
                    px(10200),
                    50,
                    vec![order(1, Side::Sell, px(10200), 50, 50)]
                )]
            )
        );
    }

    #[tokio::test]
    async fn modify_unknown_order_leaves_snapshot_unchanged() {
        let engine = spawn_engine();

        engine.submit_order(Side::Buy, 1, px(10000), 100).await;

        assert_eq!(
            engine.modify_order(999, px(10000), 20).await,
            OrderAck::Rejected {
                reason: format!("unknown order id:{}", 999),
            }
        );

        assert_eq!(
            engine.snapshot().await.bids,
            book(
                Side::Buy,
                vec![level(
                    px(10000),
                    100,
                    vec![order(1, Side::Buy, px(10000), 100, 100)]
                )]
            )
        );
    }

    #[tokio::test]
    async fn no_match_both_orders_rest() {
        let engine = spawn_engine();

        engine.submit_order(Side::Buy, 1, px(10000), 10).await;
        engine.submit_order(Side::Sell, 2, px(10100), 10).await;

        let snapshot = engine.snapshot().await;

        assert_eq!(snapshot.bids, equivalent(Side::Buy, &[(1, px(10000), 10)]));
        assert_eq!(snapshot.asks, equivalent(Side::Sell, &[(2, px(10100), 10)]));
    }

    #[tokio::test]
    async fn one_bid_matches_ask() {
        let engine = spawn_engine();

        engine.submit_order(Side::Sell, 1, px(10000), 10).await;
        engine.submit_order(Side::Buy, 2, px(10000), 10).await;

        let snapshot = engine.snapshot().await;
        assert_eq!(snapshot.bids, equivalent(Side::Buy, &[]));
        assert_eq!(snapshot.asks, equivalent(Side::Sell, &[]));
    }

    #[tokio::test]
    async fn one_ask_matches_bid() {
        let engine = spawn_engine();

        engine.submit_order(Side::Buy, 1, px(10000), 10).await;
        engine.submit_order(Side::Sell, 2, px(10000), 10).await;

        let snapshot = engine.snapshot().await;
        assert_eq!(snapshot.bids, equivalent(Side::Buy, &[]));
        assert_eq!(snapshot.asks, equivalent(Side::Sell, &[]));
    }

    #[tokio::test]
    async fn reprice_via_modify_matches() {
        let engine = spawn_engine();

        engine.submit_order(Side::Sell, 1, px(10100), 10).await;
        engine.submit_order(Side::Buy, 2, px(10000), 10).await;

        let snapshot = engine.snapshot().await;
        assert_eq!(snapshot.bids, equivalent(Side::Buy, &[(2, px(10000), 10)]));
        assert_eq!(snapshot.asks, equivalent(Side::Sell, &[(1, px(10100), 10)]));

        assert_eq!(
            engine.modify_order(2, px(10100), 10).await,
            OrderAck::Modified {
                order_id: 2,
                price: px(10100),
                qty: 10,
            }
        );

        let snapshot = engine.snapshot().await;
        assert_eq!(snapshot.bids, equivalent(Side::Buy, &[]));
        assert_eq!(snapshot.asks, equivalent(Side::Sell, &[]));
    }
}
