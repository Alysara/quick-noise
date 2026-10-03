/// Tiny module for fast psuedo-random bit mixing.
pub struct Random {}

impl Random {
    pub fn mix_u64(mut data: u64) -> u64 {
        data ^= 0xB820ABC04DB1A623;
        data ^= data >> 33;
        data = data.wrapping_mul(0xff51afd7ed558ccd);
        data ^= data >> 33;
        data = data.wrapping_mul(0xc4ceb9fe1a85ec53);
        data ^= data >> 33;
        data
    }

    pub fn mix_u64_pair(mut data1: u64, data2: u64) -> u64 {
        data1 ^= 0xB820ABC04DB1A623;
        data1 ^= data1 >> 33;
        data1 = data1.wrapping_mul(0xff51afd7ed558ccd ^ data2);
        data1 ^= data1 >> 33;
        data1 = data1.wrapping_mul(0xc4ceb9fe1a85ec53 ^ data2);
        data1 ^= data1 >> 33;
        data1
    }

    pub fn mix_u64_triple(mut data1: u64, data2: u64, data3: u64) -> u64 {
        data1 ^= 0xB820ABC04DB1A623;
        data1 ^= data1 >> 33;
        data1 = data1.wrapping_mul(0xff51afd7ed558ccd ^ data2);
        data1 ^= data1 >> 33;
        data1 = data1.wrapping_mul(0xc4ceb9fe1a85ec53 ^ data3);
        data1 ^= data1 >> 33;
        data1 = data1.wrapping_mul(0xff51afd7ed558ccd ^ data2);
        data1 ^= data1 >> 33;
        data1
    }

    // pub fn mix_u32(mut data: u32) -> u32 {
    //     data ^= 0x7A019853;
    //     data ^= data >> 16;
    //     data = data.wrapping_mul(0x85ebca6b);
    //     data ^= data >> 13;
    //     data = data.wrapping_mul(0xc2b2ae35);
    //     data ^= data >> 16;
    //     data
    // }
}
