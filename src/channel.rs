use crate::book::{Book, CancelOrder, ModifyOrder, NewOrder, Side};
use rust_decimal::Decimal;
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};

pub type OrderId = u64;
pub type RejectReason = String;
pub type CancelReason = String;

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct EngineSnapshot {
    pub bid_levels: Vec<(Decimal, i32)>,
    pub ask_levels: Vec<(Decimal, i32)>,
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
                    side_of.insert(order_id, side);
                    let book = pick(side, &mut bids, &mut asks);
                    book.submit_order(NewOrder {
                        id: order_id,
                        side,
                        price,
                        qty,
                    });
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
                        let book = pick(side, &mut bids, &mut asks);
                        book.modify_order(ModifyOrder {
                            id: order_id,
                            new_price: price,
                            new_qty: qty,
                        });
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
                        bid_levels: bids
                            .view()
                            .levels
                            .iter()
                            .map(|l| (l.price, l.total_qty))
                            .collect(),
                        ask_levels: asks
                            .view()
                            .levels
                            .iter()
                            .map(|l| (l.price, l.total_qty))
                            .collect(),
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

    fn px(cents: i64) -> Decimal {
        Decimal::new(cents, 2)
    }

    #[tokio::test]
    async fn test_submit_order() {
        let engine = spawn_engine();

        let order_id = 1;
        let side = Side::Buy;
        let price = Decimal::new(100, 2);
        let qty = 100;

        let ack = engine.submit_order(side, order_id, price, qty).await;

        assert_eq!(ack, OrderAck::Accepted { order_id });
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
            snapshot,
            EngineSnapshot {
                bid_levels: vec![(px(10000), 100)],
                ask_levels: vec![(px(10100), 50)],
            }
        );
    }

    #[tokio::test]
    async fn cancel_order_disappears_from_snapshot() {
        let engine = spawn_engine();

        engine.submit_order(Side::Buy, 1, px(10000), 100).await;
        engine.submit_order(Side::Buy, 2, px(10000), 40).await;

        // Both rest on the same bid level before cancelling.
        assert_eq!(
            engine.snapshot().await,
            EngineSnapshot {
                bid_levels: vec![(px(10000), 140)],
                ask_levels: vec![],
            }
        );

        assert_eq!(
            engine.cancel_order(1).await,
            OrderAck::Cancelled {
                order_id: 1,
                reason: format!("{} Cancelled", 1),
            }
        );

        // Only order 2's quantity remains on the level.
        assert_eq!(
            engine.snapshot().await,
            EngineSnapshot {
                bid_levels: vec![(px(10000), 40)],
                ask_levels: vec![],
            }
        );
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
            engine.snapshot().await,
            EngineSnapshot {
                bid_levels: vec![],
                ask_levels: vec![(px(10100), 50)],
            }
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
            engine.snapshot().await,
            EngineSnapshot {
                bid_levels: vec![],
                ask_levels: vec![(px(10100), 20)],
            }
        );
    }

    #[tokio::test]
    async fn modify_order_reprices_in_snapshot() {
        let engine = spawn_engine();

        engine.submit_order(Side::Sell, 1, px(10100), 50).await;

        // Repricing relocates the order to the new level.
        assert_eq!(
            engine.modify_order(1, px(10200), 50).await,
            OrderAck::Modified {
                order_id: 1,
                price: px(10200),
                qty: 50,
            }
        );

        assert_eq!(
            engine.snapshot().await,
            EngineSnapshot {
                bid_levels: vec![],
                ask_levels: vec![(px(10200), 50)],
            }
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
            engine.snapshot().await,
            EngineSnapshot {
                bid_levels: vec![(px(10000), 100)],
                ask_levels: vec![],
            }
        );
    }
}
