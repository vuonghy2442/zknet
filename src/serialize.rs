
pub trait MySerialize {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W);
    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self;
}

pub trait SimplySerialize {}

impl<T:SimplySerialize> MySerialize for T {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        let ptr = unsafe{any_as_u8_slice(self)};
        w.write(ptr).unwrap();
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        unsafe {
            let mut s: Self = std::mem::MaybeUninit::uninit().assume_init();
            r.read_exact(any_as_u8_slice_mut(&mut s)).unwrap();
            s
        }
    }
}


impl MySerialize for String {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        self.as_bytes().len().my_serialize(w);
        w.write(self.as_bytes()).unwrap();
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        let len = usize::my_deserialize(r);
        let mut buf: Vec<u8> = Vec::new();
        buf.resize(len, 0);
        r.read_exact(&mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }
}

impl<T:MySerialize> MySerialize for Option<T> {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        match self {
            Some(val) => {w.write(&[1]).unwrap(); val.my_serialize(w)},
            None => {w.write(&[0]).unwrap();},
        }
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        let mut val: [u8;1] = [0];
        r.read_exact(&mut val).unwrap();
        if val[0] == 1 {
            Some(T::my_deserialize(r))
        } else {
            None
        }
    }
}


impl SimplySerialize for usize {}
impl SimplySerialize for u32 {}
impl SimplySerialize for i32 {}
impl SimplySerialize for f64 {}

impl<T: SimplySerialize> MySerialize for Box<[T]> {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        let len = self.len();
        len.my_serialize(w);
        w.write(unsafe{any_slice_as_u8_slice(self.as_ref())}).unwrap();
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        let len = usize::my_deserialize(r);
        let mut data: Vec<T> = Vec:: with_capacity(len);
        unsafe {data.set_len(len) };
        r.read_exact(unsafe {any_slice_as_u8_slice_mut(data.as_mut_slice())}).unwrap();
        data.into_boxed_slice()
    }
}

impl<T: SimplySerialize> MySerialize for Vec<T> {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        let len = self.len();
        len.my_serialize(w);
        w.write(unsafe{any_slice_as_u8_slice(&self)}).unwrap();
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        let len = usize::my_deserialize(r);
        let mut data: Vec<T> = Vec:: with_capacity(len);
        unsafe {data.set_len(len) };
        r.read_exact(unsafe {any_slice_as_u8_slice_mut(&mut data)}).unwrap();
        data
    }
}

pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}


pub unsafe fn any_slice_as_u8_slice<T: Sized>(p: &[T]) -> &[u8] {
    ::std::slice::from_raw_parts(
        (&p[0] as *const T) as *const u8,
        ::std::mem::size_of::<T>() * p.len(),
    )
}

pub unsafe fn any_as_u8_slice_mut<T: Sized>(p: &mut T) -> &mut [u8] {
    ::std::slice::from_raw_parts_mut(
        (p as *mut T) as *mut u8,
        ::std::mem::size_of::<T>(),
    )
}

pub unsafe fn any_slice_as_u8_slice_mut<T: Sized>(p: &mut [T]) -> &mut [u8] {
    ::std::slice::from_raw_parts_mut(
        (&mut p[0] as *mut T) as *mut u8,
        ::std::mem::size_of::<T>() * p.len(),
    )
}
