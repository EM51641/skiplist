use crate::book::{Book, CancelOrder, ModifyOrder, NewOrder, Side};
use rust_decimal::Decimal;
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};

pub type OrderId = u64;
pub type RejectReason = String;
pub type CancelReason = String;

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
}

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
            }
        }
    });

    EngineHandle { tx }
}
