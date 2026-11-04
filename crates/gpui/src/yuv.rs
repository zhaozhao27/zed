use std::fmt;
use std::sync::Arc;

/// The YCbCr interpretation applied when a [`YuvFrame`] is sampled on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YuvColorSpace {
    /// BT.601 with the full 0..255 luma range.
    Bt601Full,
    /// BT.601 with the 16..235 studio range.
    Bt601Limited,
    /// BT.709 with the full 0..255 luma range.
    Bt709Full,
    /// BT.709 with the 16..235 studio range.
    Bt709Limited,
}

impl YuvColorSpace {
    /// Columns of the YCbCr-to-RGB matrix: luma, Cb, Cr, then the offset row.
    ///
    /// The shader multiplies this matrix by `vec4(y, cb, cr, 1)`.
    pub const fn matrix(self) -> [[f32; 4]; 4] {
        match self {
            Self::Bt601Full => [
                [1.0, 1.0, 1.0, 0.0],
                [0.0, -0.344136, 1.772, 0.0],
                [1.402, -0.714136, 0.0, 0.0],
                [-0.701, 0.529136, -0.886, 1.0],
            ],
            Self::Bt601Limited => [
                [1.164, 1.164, 1.164, 0.0],
                [0.0, -0.391762, 2.017232, 0.0],
                [1.596027, -0.812968, 0.0, 0.0],
                [-0.874202, 0.531668, -1.085631, 1.0],
            ],
            Self::Bt709Full => [
                [1.0, 1.0, 1.0, 0.0],
                [0.0, -0.187330, 1.8556, 0.0],
                [1.5748, -0.468124, 0.0, 0.0],
                [-0.7874, 0.327722, -0.9278, 1.0],
            ],
            Self::Bt709Limited => [
                [1.164, 1.164, 1.164, 0.0],
                [0.0, -0.213249, 2.112402, 0.0],
                [1.792741, -0.532909, 0.0, 0.0],
                [-0.972945, 0.301483, -1.133402, 1.0],
            ],
        }
    }
}

/// A planar YCbCr 4:2:0 frame in NV12 layout: a full-resolution luma plane
/// followed by one interleaved CbCr plane at half resolution in both axes.
#[derive(Clone)]
pub struct YuvFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Luma samples, row-major.
    pub y_plane: Arc<Vec<u8>>,
    /// Distance in bytes between the starts of consecutive luma rows.
    pub y_stride: u32,
    /// Interleaved CbCr sample pairs, row-major.
    pub uv_plane: Arc<Vec<u8>>,
    /// Distance in bytes between the starts of consecutive chroma rows.
    pub uv_stride: u32,
    /// How the shader should interpret the samples.
    pub color_space: YuvColorSpace,
}

impl YuvFrame {
    /// Size of the chroma plane in texels.
    pub fn chroma_size(&self) -> (u32, u32) {
        (self.width.div_ceil(2), self.height.div_ceil(2))
    }

    /// Whether every plane is large enough for the declared dimensions and stride.
    pub fn is_valid(&self) -> bool {
        if self.width == 0 || self.height == 0 {
            return false;
        }
        let luma_rows = self.y_stride as u64 * self.height as u64;
        let (chroma_width, chroma_height) = self.chroma_size();
        let chroma_rows = self.uv_stride as u64 * chroma_height as u64;
        self.y_stride as u64 >= self.width as u64
            && self.uv_stride as u64 >= chroma_width as u64 * 2
            && self.y_plane.len() as u64 >= luma_rows
            && self.uv_plane.len() as u64 >= chroma_rows
    }
}

impl fmt::Debug for YuvFrame {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("YuvFrame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("y_stride", &self.y_stride)
            .field("uv_stride", &self.uv_stride)
            .field("color_space", &self.color_space)
            .finish_non_exhaustive()
    }
}

impl PartialEq for YuvFrame {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.y_stride == other.y_stride
            && self.uv_stride == other.uv_stride
            && self.color_space == other.color_space
            && Arc::ptr_eq(&self.y_plane, &other.y_plane)
            && Arc::ptr_eq(&self.uv_plane, &other.uv_plane)
    }
}

impl Eq for YuvFrame {}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(color_space: YuvColorSpace, y: f32, cb: f32, cr: f32) -> [f32; 3] {
        let matrix = color_space.matrix();
        let mut out = [0.0; 3];
        for (row, value) in out.iter_mut().enumerate() {
            *value =
                matrix[0][row] * y + matrix[1][row] * cb + matrix[2][row] * cr + matrix[3][row];
        }
        out
    }

    #[test]
    fn limited_range_white_and_black_map_to_the_unit_range() {
        for color_space in [YuvColorSpace::Bt601Limited, YuvColorSpace::Bt709Limited] {
            let white = convert(color_space, 235.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0);
            let black = convert(color_space, 16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0);
            assert!(
                (white[0] - 1.0).abs() < 0.01,
                "{color_space:?} white {white:?}"
            );
            assert!(
                (white[1] - 1.0).abs() < 0.01,
                "{color_space:?} white {white:?}"
            );
            assert!(
                (white[2] - 1.0).abs() < 0.01,
                "{color_space:?} white {white:?}"
            );
            assert!(black[0].abs() < 0.01, "{color_space:?} black {black:?}");
            assert!(black[1].abs() < 0.01, "{color_space:?} black {black:?}");
            assert!(black[2].abs() < 0.01, "{color_space:?} black {black:?}");
        }
    }

    #[test]
    fn validation_requires_complete_planes() {
        let valid = YuvFrame {
            width: 4,
            height: 4,
            y_plane: Arc::new(vec![0; 4 * 4]),
            y_stride: 4,
            uv_plane: Arc::new(vec![0; 4 * 2]),
            uv_stride: 4,
            color_space: YuvColorSpace::Bt601Limited,
        };
        assert!(valid.is_valid());

        let mut short = valid.clone();
        short.uv_plane = Arc::new(vec![0; 3]);
        assert!(!short.is_valid());

        let mut narrow_stride = valid.clone();
        narrow_stride.y_stride = 2;
        assert!(!narrow_stride.is_valid());
    }

    #[test]
    fn odd_dimensions_round_the_chroma_plane_up() {
        let frame = YuvFrame {
            width: 5,
            height: 3,
            y_plane: Arc::new(vec![0; 5 * 3]),
            y_stride: 5,
            uv_plane: Arc::new(vec![0; 6 * 2]),
            uv_stride: 6,
            color_space: YuvColorSpace::Bt601Limited,
        };
        assert_eq!(frame.chroma_size(), (3, 2));
        assert!(frame.is_valid());
    }
}
