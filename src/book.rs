use crate::order::{CancelOrder, ModifyOrder, NewOrder, OrderId, Side};
use crate::skiplist::SkipList;
use crate::slab::{NodeId, OrderNode, Slab};
use rust_decimal::Decimal;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Level {
    pub head: Option<NodeId>,
    pub tail: Option<NodeId>,
    pub total_qty: i32,
}

pub struct Book {
    side: Side,
    slab: Slab,
    order_index: HashMap<OrderId, NodeId>,
    levels: SkipList<Decimal, Level>,
}

impl Book {
    pub fn new(
        side: Side,
        levels: Option<SkipList<Decimal, Level>>,
        slab: Option<Slab>,
        order_index: Option<HashMap<OrderId, NodeId>>,
    ) -> Self {
        let levels = levels.unwrap_or_else(|| SkipList::new(16));
        let slab = slab.unwrap_or_else(|| Slab::new());
        let order_index = order_index.unwrap_or_else(|| HashMap::new());

        Self {
            side,
            slab,
            order_index,
            levels,
        }
    }

    fn key(&self, price: Decimal) -> Decimal {
        match self.side {
            Side::Sell => price,
            Side::Buy => -price,
        }
    }

    pub fn side(&self) -> Side {
        self.side
    }

    pub fn levels(&self) -> &SkipList<Decimal, Level> {
        &self.levels
    }

    pub fn slab(&self) -> &Slab {
        &self.slab
    }

    /// Price and remaining quantity of a resting order, if it is still on the book.
    pub fn resting_order(&self, id: OrderId) -> Option<(Decimal, i32)> {
        let nid = *self.order_index.get(&id)?;
        let node = self.slab.get(nid)?;
        Some((node.price, node.remaining))
    }

    pub fn submit_order(&mut self, req: NewOrder) {
        let nid = self.slab.insert(OrderNode {
            id: req.id,
            side: req.side,
            price: req.price,
            qty: req.qty,
            remaining: req.qty,
            prev: None,
            next: None,
        });

        let k = self.key(req.price);
        match self.levels.get_mut(&k) {
            Some(level) => {
                let prev_tail = level.tail;
                level.tail = Some(nid);
                level.total_qty += req.qty;
                if let Some(t) = prev_tail {
                    self.slab.get_mut(t).unwrap().next = Some(nid);
                    self.slab.get_mut(nid).unwrap().prev = Some(t);
                }
            }
            None => {
                self.levels.insert(
                    k,
                    Level {
                        head: Some(nid),
                        tail: Some(nid),
                        total_qty: req.qty,
                    },
                );
            }
        }

        self.order_index.insert(req.id, nid);
    }

    pub fn cancel_order(&mut self, req: CancelOrder) -> Option<OrderNode> {
        let nid = self.order_index.remove(&req.id)?;
        let (price, remaining, prev, next) = {
            let n = self.slab.get(nid)?;
            (n.price, n.remaining, n.prev, n.next)
        };

        if let Some(p) = prev {
            self.slab.get_mut(p).unwrap().next = next;
        }
        if let Some(n) = next {
            self.slab.get_mut(n).unwrap().prev = prev;
        }

        let k = self.key(price);
        let level_empty = match self.levels.get_mut(&k) {
            Some(level) => {
                if level.head == Some(nid) {
                    level.head = next;
                }
                if level.tail == Some(nid) {
                    level.tail = prev;
                }
                level.total_qty -= remaining;
                level.head.is_none()
            }
            None => false,
        };
        if level_empty {
            self.levels.remove(&k);
        }

        self.slab.remove(nid)
    }

    pub fn modify_order(&mut self, req: ModifyOrder) {
        let nid = *self.order_index.get(&req.id).unwrap();
        let (old_price, old_remaining, side) = {
            let n = self.slab.get(nid).unwrap();
            (n.price, n.remaining, n.side)
        };

        let price_changed = req.new_price != old_price;
        let qty_increased = req.new_qty > old_remaining;

        if price_changed || qty_increased {
            self.cancel_order(CancelOrder { id: req.id });
            self.submit_order(NewOrder {
                id: req.id,
                side,
                price: req.new_price,
                qty: req.new_qty,
            });
        } else {
            self.slab.get_mut(nid).unwrap().remaining = req.new_qty;
            let k = self.key(old_price);
            if let Some(level) = self.levels.get_mut(&k) {
                level.total_qty += req.new_qty - old_remaining;
            }
        }
    }

    /// Match an incoming order of `target_qty` priced at `target_price` against
    /// this (opposite-side) resting book, consuming crossing levels in
    /// price-time priority. Returns the quantity left unfilled, which the caller
    /// should rest on its own side of the book.
    pub fn book_matcher(&mut self, target_price: &Decimal, mut target_qty: i32) -> i32 {
        match self.side {
            Side::Buy => {
                // Incoming sell crosses bids priced at or above `target_price`,
                // highest bid first (the front of the skiplist for a bid book).
                while let Some((price, _)) = self.levels.first()
                    && -price >= *target_price
                    && target_qty > 0
                {
                    let price = *price;
                    self.consume_level(price, &mut target_qty);
                }
            }
            Side::Sell => {
                // Incoming buy crosses asks priced at or below `target_price`,
                // lowest ask first (the front of the skiplist for an ask book).
                while let Some((price, _)) = self.levels.first()
                    && price <= target_price
                    && target_qty > 0
                {
                    let price = *price;
                    self.consume_level(price, &mut target_qty);
                }
            }
        }
        target_qty
    }

    fn consume_level(&mut self, price: Decimal, target_qty: &mut i32) {
        let level = self.levels.get_mut(&price).unwrap();

        while *target_qty > 0 {
            let Some(head_id) = level.head else { break };
            let (order_id, next_id, remaining) = {
                let order = self.slab.get(head_id).unwrap();
                (order.id, order.next, order.remaining)
            };

            if remaining > *target_qty {
                // Partial fill: the head stays resting with less quantity.
                self.slab.get_mut(head_id).unwrap().remaining -= *target_qty;
                level.total_qty -= *target_qty;
                *target_qty = 0;
            } else {
                // Full fill: drop the head and advance to the next order.
                *target_qty -= remaining;
                level.total_qty -= remaining;
                level.head = next_id;
                if next_id.is_none() {
                    level.tail = None;
                }
                self.slab.remove(head_id);
                self.order_index.remove(&order_id);
            }
        }

        if level.head.is_none() {
            self.levels.remove(&price);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{BookView, PriceLevelView};
    use rust_decimal::Decimal;

    fn px(cents: i64) -> Decimal {
        Decimal::new(cents, 2)
    }

    mod submit {
        use super::*;

        #[test]
        fn into_empty_level_creates_isolated_node() {
            let side = Side::Sell;
            let price = px(10000);

            let mut book = Book::new(side, None, None, None);
            book.submit_order(NewOrder {
                id: 1,
                side,
                price,
                qty: 10,
            });

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, level.tail);
            assert!(level.head.is_some());
            assert_eq!(level.total_qty, 10);
        }

        #[test]
        fn bid_keys_level_by_negated_price() {
            // The buy side inverts the price key so the highest bid sorts first,
            // so a bid level lives under `-price`, not the raw price.
            let side = Side::Buy;
            let price = px(10000);

            let mut book = Book::new(side, None, None, None);
            book.submit_order(NewOrder {
                id: 1,
                side,
                price,
                qty: 10,
            });

            assert!(book.levels.get(&price).is_none());
            let level = book.levels.get(&-price).unwrap();
            assert_eq!(level.total_qty, 10);
            assert!(level.head.is_some());
        }

        #[test]
        fn appends_to_existing_level() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid_a = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });
            let nid_b = slab.insert(OrderNode {
                id: 2,
                side,
                price,
                qty: 20,
                remaining: 20,
                prev: Some(nid_a),
                next: None,
            });
            slab.get_mut(nid_a).unwrap().next = Some(nid_b);

            let mut order_index = HashMap::new();
            order_index.insert(1, nid_a);
            order_index.insert(2, nid_b);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid_a),
                    tail: Some(nid_b),
                    total_qty: 30,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            book.submit_order(NewOrder {
                id: 3,
                side,
                price,
                qty: 30,
            });

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, Some(nid_a));
            assert_ne!(level.tail, Some(nid_b), "tail must advance past prior tail");
            assert_eq!(level.total_qty, 60);
        }

        #[test]
        fn bid_appends_to_existing_level() {
            // Same append behavior on the buy side, where the level is keyed by
            // the negated price.
            let side = Side::Buy;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid_a = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });
            let nid_b = slab.insert(OrderNode {
                id: 2,
                side,
                price,
                qty: 20,
                remaining: 20,
                prev: Some(nid_a),
                next: None,
            });
            slab.get_mut(nid_a).unwrap().next = Some(nid_b);

            let mut order_index = HashMap::new();
            order_index.insert(1, nid_a);
            order_index.insert(2, nid_b);

            let mut levels = SkipList::new(16);
            levels.insert(
                -price,
                Level {
                    head: Some(nid_a),
                    tail: Some(nid_b),
                    total_qty: 30,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            book.submit_order(NewOrder {
                id: 3,
                side,
                price,
                qty: 30,
            });

            let level = book.levels.get(&-price).unwrap();
            assert_eq!(level.head, Some(nid_a));
            assert_ne!(level.tail, Some(nid_b), "tail must advance past prior tail");
            assert_eq!(level.total_qty, 60);
        }
    }

    mod cancel {
        use super::*;

        #[test]
        fn only_order_removes_level() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });

            let mut order_index = HashMap::new();
            order_index.insert(1, nid);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid),
                    tail: Some(nid),
                    total_qty: 10,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            let removed = book.cancel_order(CancelOrder { id: 1 });

            assert!(removed.is_some());
            assert_eq!(removed.unwrap().id, 1);
            assert!(book.order_index.is_empty());
            assert!(book.levels.get(&price).is_none());
        }

        #[test]
        fn head_relinks_and_updates_level() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid_a = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });
            let nid_b = slab.insert(OrderNode {
                id: 2,
                side,
                price,
                qty: 20,
                remaining: 20,
                prev: Some(nid_a),
                next: None,
            });
            let nid_c = slab.insert(OrderNode {
                id: 3,
                side,
                price,
                qty: 30,
                remaining: 30,
                prev: Some(nid_b),
                next: None,
            });
            slab.get_mut(nid_a).unwrap().next = Some(nid_b);
            slab.get_mut(nid_b).unwrap().next = Some(nid_c);

            let mut order_index = HashMap::new();
            order_index.insert(1, nid_a);
            order_index.insert(2, nid_b);
            order_index.insert(3, nid_c);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid_a),
                    tail: Some(nid_c),
                    total_qty: 60,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            book.cancel_order(CancelOrder { id: 1 });

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, Some(nid_b));
            assert_eq!(level.tail, Some(nid_c));
            assert_eq!(level.total_qty, 50);
        }

        #[test]
        fn tail_relinks_and_updates_level() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid_a = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });
            let nid_b = slab.insert(OrderNode {
                id: 2,
                side,
                price,
                qty: 20,
                remaining: 20,
                prev: Some(nid_a),
                next: None,
            });
            let nid_c = slab.insert(OrderNode {
                id: 3,
                side,
                price,
                qty: 30,
                remaining: 30,
                prev: Some(nid_b),
                next: None,
            });
            slab.get_mut(nid_a).unwrap().next = Some(nid_b);
            slab.get_mut(nid_b).unwrap().next = Some(nid_c);

            let mut order_index = HashMap::new();
            order_index.insert(1, nid_a);
            order_index.insert(2, nid_b);
            order_index.insert(3, nid_c);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid_a),
                    tail: Some(nid_c),
                    total_qty: 60,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            book.cancel_order(CancelOrder { id: 3 });

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, Some(nid_a));
            assert_eq!(level.tail, Some(nid_b));
            assert_eq!(level.total_qty, 30);
        }

        #[test]
        fn middle_relinks_neighbors_and_updates_level() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid_a = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });
            let nid_b = slab.insert(OrderNode {
                id: 2,
                side,
                price,
                qty: 20,
                remaining: 20,
                prev: Some(nid_a),
                next: None,
            });
            let nid_c = slab.insert(OrderNode {
                id: 3,
                side,
                price,
                qty: 30,
                remaining: 30,
                prev: Some(nid_b),
                next: None,
            });
            slab.get_mut(nid_a).unwrap().next = Some(nid_b);
            slab.get_mut(nid_b).unwrap().next = Some(nid_c);

            let mut order_index = HashMap::new();
            order_index.insert(1, nid_a);
            order_index.insert(2, nid_b);
            order_index.insert(3, nid_c);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid_a),
                    tail: Some(nid_c),
                    total_qty: 60,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            book.cancel_order(CancelOrder { id: 2 });

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, Some(nid_a));
            assert_eq!(level.tail, Some(nid_c));
            assert_eq!(level.total_qty, 40);
        }
    }

    mod modify {
        use super::*;

        fn order_ids(level: &PriceLevelView) -> Vec<OrderId> {
            level.orders.iter().map(|o| o.id).collect()
        }

        #[test]
        fn qty_decrease_keeps_priority_in_place() {
            let side = Side::Sell;
            let price = px(10000);

            let mut book = Book::new(side, None, None, None);
            book.submit_order(NewOrder {
                id: 1,
                side,
                price,
                qty: 10,
            });
            book.submit_order(NewOrder {
                id: 2,
                side,
                price,
                qty: 20,
            });
            book.submit_order(NewOrder {
                id: 3,
                side,
                price,
                qty: 30,
            });

            book.modify_order(ModifyOrder {
                id: 2,
                new_price: price,
                new_qty: 5,
            });

            let view = BookView::from_book(&book);
            assert_eq!(view.levels.len(), 1);
            let level = &view.levels[0];
            assert_eq!(order_ids(level), vec![1, 2, 3], "order 2 stays in place");
            assert_eq!(level.total_qty, 45);
            assert_eq!(level.orders[1].remaining, 5);
        }

        #[test]
        fn qty_increase_moves_to_tail() {
            let side = Side::Sell;
            let price = px(10000);

            let mut book = Book::new(side, None, None, None);
            book.submit_order(NewOrder {
                id: 1,
                side,
                price,
                qty: 10,
            });
            book.submit_order(NewOrder {
                id: 2,
                side,
                price,
                qty: 20,
            });
            book.submit_order(NewOrder {
                id: 3,
                side,
                price,
                qty: 30,
            });

            book.modify_order(ModifyOrder {
                id: 2,
                new_price: price,
                new_qty: 50,
            });

            let view = BookView::from_book(&book);
            assert_eq!(view.levels.len(), 1);
            let level = &view.levels[0];
            assert_eq!(order_ids(level), vec![1, 3, 2], "order 2 moves to the tail");
            assert_eq!(level.total_qty, 90);
            assert_eq!(level.orders[2].remaining, 50);
        }

        #[test]
        fn price_change_relocates_to_new_level_tail() {
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

            book.modify_order(ModifyOrder {
                id: 2,
                new_price: px(10100),
                new_qty: 20,
            });

            let view = BookView::from_book(&book);
            assert_eq!(view.levels.len(), 2);

            let level_10000 = &view.levels[0];
            assert_eq!(level_10000.price, px(10000));
            assert_eq!(order_ids(level_10000), vec![1]);
            assert_eq!(level_10000.total_qty, 10);

            let level_10100 = &view.levels[1];
            assert_eq!(level_10100.price, px(10100));
            assert_eq!(
                order_ids(level_10100),
                vec![3, 2],
                "order 2 joins at the tail"
            );
            assert_eq!(level_10100.total_qty, 50);
            assert_eq!(level_10100.orders[1].price, px(10100));
        }

        #[test]
        fn bid_price_change_relocates_to_new_level_tail() {
            // Same reprice-relocation on the buy side. The view reports positive
            // prices with the best (highest) bid first.
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

            book.modify_order(ModifyOrder {
                id: 2,
                new_price: px(9900),
                new_qty: 20,
            });

            let view = BookView::from_book(&book);
            assert_eq!(view.levels.len(), 2);

            let level_10000 = &view.levels[0];
            assert_eq!(level_10000.price, px(10000));
            assert_eq!(order_ids(level_10000), vec![1]);
            assert_eq!(level_10000.total_qty, 10);

            let level_9900 = &view.levels[1];
            assert_eq!(level_9900.price, px(9900));
            assert_eq!(
                order_ids(level_9900),
                vec![3, 2],
                "order 2 joins at the tail"
            );
            assert_eq!(level_9900.total_qty, 50);
            assert_eq!(level_9900.orders[1].price, px(9900));
        }
    }
}
