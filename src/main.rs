use matching_engine::skiplist::SkipList;

fn main() {
    let mut sl = SkipList::new(16);

    sl.insert(3, "three");
    sl.insert(1, "one");
    sl.insert(5, "five");
    sl.insert(2, "two");
    sl.insert(4, "four");

    println!("len = {}", sl.len());
    println!("get(3) = {:?}", sl.get(&3));
    println!("first  = {:?}", sl.first());
    println!("last   = {:?}", sl.last());

    println!("\nAll entries (sorted):");
    for (k, v) in sl.iter() {
        println!("  {k} -> {v}");
    }

    println!("\nRange [2, 4):");
    for (k, v) in sl.range(&2, &4) {
        println!("  {k} -> {v}");
    }

    sl.remove(&3);
    println!("\nAfter removing 3, len = {}", sl.len());
}
