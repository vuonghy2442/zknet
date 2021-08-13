

pub fn all_same<T: PartialEq>(arr: &[T]) -> bool {
    if arr.len() == 0 {
        return true;
    }
    for i in arr[1..].iter() {
        if *i != arr[0] {
            return false;
        }
    }
    return true;
}