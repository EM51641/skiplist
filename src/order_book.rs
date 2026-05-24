use crate::skiplist::SkipList;
use rust_decimal::Decimal;

pub struct Book {
    skiplist: SkipList<Decimal, i32>,
}

impl Book {
    pub fn new() -> Self {
        Self {
            skiplist: SkipList::new(16),
        }
    }
    pub fn add(&mut self, price: Decimal, quantity: i32) {
        match self.skiplist.get_mut(&price) {
            Some(k) => {
                *k += quantity;
            }
            None => {
                self.skiplist.insert(price, quantity);
            }
        }
    }

    pub fn remove(&mut self, price: Decimal, quantity: i32) {
        match self.skiplist.get_mut(&price) {
            Some(k) if *k > quantity => {
                *k -= quantity;
            }
            Some(_) => {
                self.skiplist.remove(&price);
            }
            None => {}
        }
    }
}
