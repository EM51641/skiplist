use crate::book::Book;
use crate::order::{OrderId, Side};
use rust_decimal::Decimal;

#[derive(Debug, Clone, PartialEq)]
pub struct BookView {
    pub side: Side,
    pub levels: Vec<PriceLevelView>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PriceLevelView {
    pub price: Decimal,
    pub total_qty: i32,
    pub orders: Vec<OrderView>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderView {
    pub id: OrderId,
    pub side: Side,
    pub price: Decimal,
    pub qty: i32,
    pub remaining: i32,
}

impl BookView {
    pub fn from_book(book: &Book) -> BookView {
        let mut levels = Vec::new();
        for (key_price, level) in book.levels().iter() {
            let mut orders = Vec::new();
            let mut current = level.head;
            while let Some(nid) = current {
                let node = book.slab().get(nid).unwrap();
                orders.push(OrderView {
                    id: node.id,
                    side: node.side,
                    price: node.price,
                    qty: node.qty,
                    remaining: node.remaining,
                });
                current = node.next;
            }
            levels.push(PriceLevelView {
                price: match book.side() {
                    Side::Buy => -*key_price,
                    Side::Sell => *key_price,
                },
                total_qty: level.total_qty,
                orders,
            });
        }
        BookView {
            side: book.side(),
            levels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order::{CancelOrder, ModifyOrder, NewOrder};
    use rust_decimal::Decimal;

    fn px(cents: i64) -> Decimal {
        Decimal::new(cents, 2)
    }

    fn submit(book: &mut Book, id: OrderId, price: Decimal, qty: i32) {
        let side = book.side();
        book.submit_order(NewOrder {
            id,
            side,
            price,
            qty,
        });
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

    // A populated ask book and a populated bid book each render their levels in
    // book order (best price first) and their orders in FIFO (time) order.
    #[test]
    fn populated_books_render_in_price_time_order() {
        let mut asks = Book::new(Side::Sell, None, None, None);
        submit(&mut asks, 1, px(10100), 5); // worse ask, out of price order on purpose
        submit(&mut asks, 2, px(10000), 10); // best ask
        submit(&mut asks, 3, px(10000), 7); // same level, later in time

        assert_eq!(
            BookView::from_book(&asks),
            BookView {
                side: Side::Sell,
                levels: vec![
                    PriceLevelView {
                        price: px(10000),
                        total_qty: 17,
                        orders: vec![
                            order(2, Side::Sell, px(10000), 10, 10),
                            order(3, Side::Sell, px(10000), 7, 7),
                        ],
                    },
                    PriceLevelView {
                        price: px(10100),
                        total_qty: 5,
                        orders: vec![order(1, Side::Sell, px(10100), 5, 5)],
                    },
                ],
            }
        );

        let mut bids = Book::new(Side::Buy, None, None, None);
        submit(&mut bids, 10, px(9900), 8);
        submit(&mut bids, 11, px(10000), 12);
        submit(&mut bids, 12, px(9900), 4);

        assert_eq!(
            BookView::from_book(&bids),
            BookView {
                side: Side::Buy,
                levels: vec![
                    PriceLevelView {
                        price: px(10000),
                        total_qty: 12,
                        orders: vec![order(11, Side::Buy, px(10000), 12, 12)],
                    },
                    PriceLevelView {
                        price: px(9900),
                        total_qty: 12,
                        orders: vec![
                            order(10, Side::Buy, px(9900), 8, 8),
                            order(12, Side::Buy, px(9900), 4, 4),
                        ],
                    },
                ],
            }
        );
    }

    #[test]
    fn adding_an_ask_appears_in_the_view() {
        let mut asks = Book::new(Side::Sell, None, None, None);
        submit(&mut asks, 1, px(10000), 10);

        submit(&mut asks, 2, px(10100), 5);

        assert_eq!(
            BookView::from_book(&asks),
            BookView {
                side: Side::Sell,
                levels: vec![
                    PriceLevelView {
                        price: px(10000),
                        total_qty: 10,
                        orders: vec![order(1, Side::Sell, px(10000), 10, 10)],
                    },
                    PriceLevelView {
                        price: px(10100),
                        total_qty: 5,
                        orders: vec![order(2, Side::Sell, px(10100), 5, 5)],
                    },
                ],
            }
        );
    }

    #[test]
    fn adding_a_bid_appears_in_the_view() {
        let mut bids = Book::new(Side::Buy, None, None, None);
        submit(&mut bids, 1, px(10000), 10);
        submit(&mut bids, 2, px(9900), 5);

        assert_eq!(
            BookView::from_book(&bids),
            BookView {
                side: Side::Buy,
                levels: vec![
                    PriceLevelView {
                        price: px(10000),
                        total_qty: 10,
                        orders: vec![order(1, Side::Buy, px(10000), 10, 10)],
                    },
                    PriceLevelView {
                        price: px(9900),
                        total_qty: 5,
                        orders: vec![order(2, Side::Buy, px(9900), 5, 5)],
                    },
                ],
            }
        );
    }

    mod ask {
        use super::*;

        #[test]
        fn insert() {
            let side = Side::Sell;

            let mut book = Book::new(side, None, None, None);
            // Submit out of price order to prove the view sorts ascending.
            book.submit_order(NewOrder {
                id: 1,
                side,
                price: px(10100),
                qty: 20,
            });
            book.submit_order(NewOrder {
                id: 2,
                side,
                price: px(10000),
                qty: 10,
            });
            book.submit_order(NewOrder {
                id: 3,
                side,
                price: px(10000),
                qty: 30,
            });

            let view = BookView::from_book(&book);

            assert_eq!(
                view,
                BookView {
                    side,
                    levels: vec![
                        PriceLevelView {
                            price: px(10000),
                            total_qty: 40,
                            orders: vec![
                                OrderView {
                                    id: 2,
                                    side,
                                    price: px(10000),
                                    qty: 10,
                                    remaining: 10,
                                },
                                OrderView {
                                    id: 3,
                                    side,
                                    price: px(10000),
                                    qty: 30,
                                    remaining: 30,
                                },
                            ],
                        },
                        PriceLevelView {
                            price: px(10100),
                            total_qty: 20,
                            orders: vec![OrderView {
                                id: 1,
                                side,
                                price: px(10100),
                                qty: 20,
                                remaining: 20,
                            }],
                        },
                    ],
                }
            );
        }

        #[test]
        fn delete() {
            let side = Side::Sell;

            let mut book = Book::new(side, None, None, None);
            book.submit_order(NewOrder {
                id: 1,
                side,
                price: px(10000),
                qty: 10,
            });
            book.submit_order(NewOrder {
                id: 2,
                side,
                price: px(10000),
                qty: 20,
            });
            book.submit_order(NewOrder {
                id: 3,
                side,
                price: px(10100),
                qty: 30,
            });

            book.cancel_order(CancelOrder { id: 1 });

            let view = BookView::from_book(&book);

            assert_eq!(
                view,
                BookView {
                    side,
                    levels: vec![
                        PriceLevelView {
                            price: px(10000),
                            total_qty: 20,
                            orders: vec![OrderView {
                                id: 2,
                                side,
                                price: px(10000),
                                qty: 20,
                                remaining: 20,
                            }],
                        },
                        PriceLevelView {
                            price: px(10100),
                            total_qty: 30,
                            orders: vec![OrderView {
                                id: 3,
                                side,
                                price: px(10100),
                                qty: 30,
                                remaining: 30,
                            }],
                        },
                    ],
                }
            );
        }

        #[test]
        fn modify() {
            let side = Side::Sell;

            let mut book = Book::new(side, None, None, None);
            book.submit_order(NewOrder {
                id: 1,
                side,
                price: px(10000),
                qty: 10,
            });
            book.submit_order(NewOrder {
                id: 2,
                side,
                price: px(10000),
                qty: 20,
            });

            book.modify_order(ModifyOrder {
                id: 2,
                new_price: px(10000),
                new_qty: 5,
            });

            let view = BookView::from_book(&book);

            assert_eq!(
                view,
                BookView {
                    side,
                    levels: vec![PriceLevelView {
                        price: px(10000),
                        total_qty: 15,
                        orders: vec![
                            OrderView {
                                id: 1,
                                side,
                                price: px(10000),
                                qty: 10,
                                remaining: 10,
                            },
                            OrderView {
                                id: 2,
                                side,
                                price: px(10000),
                                qty: 20,
                                remaining: 5,
                            },
                        ],
                    }],
                }
            );
        }
    }

    mod bid {
        use super::*;

        #[test]
        fn insert() {
            let side = Side::Buy;

            let mut book = Book::new(side, None, None, None);
            book.submit_order(NewOrder {
                id: 1,
                side,
                price: px(9900),
                qty: 20,
            });
            book.submit_order(NewOrder {
                id: 2,
                side,
                price: px(10000),
                qty: 10,
            });
            book.submit_order(NewOrder {
                id: 3,
                side,
                price: px(10000),
                qty: 30,
            });

            let view = BookView::from_book(&book);

            // Best bid (highest price) first; prices reported as positive values.
            assert_eq!(
                view,
                BookView {
                    side,
                    levels: vec![
                        PriceLevelView {
                            price: px(10000),
                            total_qty: 40,
                            orders: vec![
                                OrderView {
                                    id: 2,
                                    side,
                                    price: px(10000),
                                    qty: 10,
                                    remaining: 10,
                                },
                                OrderView {
                                    id: 3,
                                    side,
                                    price: px(10000),
                                    qty: 30,
                                    remaining: 30,
                                },
                            ],
                        },
                        PriceLevelView {
                            price: px(9900),
                            total_qty: 20,
                            orders: vec![OrderView {
                                id: 1,
                                side,
                                price: px(9900),
                                qty: 20,
                                remaining: 20,
                            }],
                        },
                    ],
                }
            );
        }

        #[test]
        fn delete() {
            let side = Side::Buy;

            let mut book = Book::new(side, None, None, None);
            book.submit_order(NewOrder {
                id: 1,
                side,
                price: px(10000),
                qty: 10,
            });
            book.submit_order(NewOrder {
                id: 2,
                side,
                price: px(10000),
                qty: 20,
            });
            book.submit_order(NewOrder {
                id: 3,
                side,
                price: px(9900),
                qty: 30,
            });

            book.cancel_order(CancelOrder { id: 1 });

            let view = BookView::from_book(&book);

            assert_eq!(
                view,
                BookView {
                    side,
                    levels: vec![
                        PriceLevelView {
                            price: px(10000),
                            total_qty: 20,
                            orders: vec![OrderView {
                                id: 2,
                                side,
                                price: px(10000),
                                qty: 20,
                                remaining: 20,
                            }],
                        },
                        PriceLevelView {
                            price: px(9900),
                            total_qty: 30,
                            orders: vec![OrderView {
                                id: 3,
                                side,
                                price: px(9900),
                                qty: 30,
                                remaining: 30,
                            }],
                        },
                    ],
                }
            );
        }

        #[test]
        fn modify() {
            let side = Side::Buy;

            let mut book = Book::new(side, None, None, None);
            book.submit_order(NewOrder {
                id: 1,
                side,
                price: px(10000),
                qty: 10,
            });
            book.submit_order(NewOrder {
                id: 2,
                side,
                price: px(10000),
                qty: 20,
            });

            book.modify_order(ModifyOrder {
                id: 2,
                new_price: px(10000),
                new_qty: 5,
            });

            let view = BookView::from_book(&book);

            assert_eq!(
                view,
                BookView {
                    side,
                    levels: vec![PriceLevelView {
                        price: px(10000),
                        total_qty: 15,
                        orders: vec![
                            OrderView {
                                id: 1,
                                side,
                                price: px(10000),
                                qty: 10,
                                remaining: 10,
                            },
                            OrderView {
                                id: 2,
                                side,
                                price: px(10000),
                                qty: 20,
                                remaining: 5,
                            },
                        ],
                    }],
                }
            );
        }
    }
}
