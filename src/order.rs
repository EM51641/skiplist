use rust_decimal::Decimal;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

pub type OrderId = u64;

pub struct NewOrder {
    pub id: OrderId,
    pub side: Side,
    pub price: Decimal,
    pub qty: i32,
}

pub struct CancelOrder {
    pub id: OrderId,
}

pub struct ModifyOrder {
    pub id: OrderId,
    pub new_price: Decimal,
    pub new_qty: i32,
}
