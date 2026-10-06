// Hash constants.
pub(crate) const BYTE_SHUFFLE: [u8; 64] = [
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13,
];

pub(crate) const HASH_PRIME: u32 = 0x85ebca6b;
pub(crate) const HASH_MASK: u32 = 0x007FFFFF;
pub(crate) const VALUE_EXP_MASK: u32 = 0x40000000;
pub(crate) const CELLULAR_EXP_MASK: u32 = 0x3F800000;

// 3D Gradients are encoded into a table:
//
//               0    1     2    3
// Dictionary: [0.0, 1.0, -1.0, 0.0]
//
// Table:
//   X     Y     Z            X  Y  Z
// [ 1.0,   1.0,  0.0]   ->   1, 1, 0
// [-1.0,   1.0,  0.0]   ->   2, 1, 0
// [ 1.0,  -1.0,  0.0]   ->   1, 2, 0
// [-1.0,  -1.0,  0.0]   ->   2, 2, 0
// [ 1.0,   0.0,  1.0]   ->   1, 0, 1
// [-1.0,   0.0,  1.0]   ->   2, 0, 1
// [ 1.0,   0.0, -1.0]   ->   1, 0, 2
// [-1.0,   0.0, -1.0]   ->   2, 0, 2
// [ 0.0,   1.0,  1.0]   ->   0, 1, 1
// [ 0.0,  -1.0,  1.0]   ->   0, 2, 1
// [ 0.0,   1.0, -1.0]   ->   0, 1, 2
// [ 0.0,  -1.0, -1.0]   ->   0, 2, 2
// [ 1.0,   1.0,  0.0]   ->   1, 1, 0
// [-1.0,   1.0,  0.0]   ->   2, 1, 0
// [ 0.0,  -1.0,  1.0]   ->   0, 2, 1
// [ 0.0,  -1.0, -1.0]   ->   0, 2, 2
//
// Encode table as a series of 2 bit indices into the dictionary:
//              <-------- Right to left
// X: 10 01 00 00 10 10 01 01 10 10 01 01 00 00 00 00
// Y: 10 10 01 01 10 01 10 01 00 00 00 00 10 10 01 01
// Z: 00 00 10 01 00 00 00 00 10 01 10 01 10 01 10 01
//
// Final Encodings:
// X: 90A5A500
// Y: A59900A5
// Z: 09009999

pub(crate) const GRAD_TABLE: [f32; 4] = [0.0, 1.0, -1.0, 0.0];

pub(crate) const X_GRAD_ENCODING: u32 = 0x90A5A500;
pub(crate) const Y_GRAD_ENCODING: u32 = 0xA59900A5;
pub(crate) const Z_GRAD_ENCODING: u32 = 0x09009999;

