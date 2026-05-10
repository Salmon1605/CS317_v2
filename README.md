# Augmentation Pipeline cho Segmentation Robustness Analysis

Tài liệu này mô tả chi tiết pipeline augmentation trong [hazelnut_ad/augmentations.py](hazelnut_ad/augmentations.py). Pipeline được thiết kế cho nghiên cứu robustness của bài toán segmentation, mô phỏng corruptions thực tế và hỗ trợ đánh giá các mô hình như FastSAM và MobileSAM.

---

## 1. Tổng quan dự án

### Mục tiêu

Pipeline augmentation hướng tới:

- Tạo tập dữ liệu mở rộng có tính đa dạng cao về hình học và nhiễu.
- Mô phỏng các dạng corruption thường gặp trong thực tế (noise, JPEG artifacts, color space shift).
- Chuẩn hoá quy trình tạo dữ liệu phục vụ robustness analysis và benchmarking.

### Vai trò của augmentation trong computer vision

- Tăng độ đa dạng của dữ liệu, giảm overfitting.
- Tạo ra các biến thể có phân phối gần với môi trường thực tế.
- Cho phép đánh giá độ ổn định của mô hình trước các biến đổi ảnh.

### Vai trò trong image segmentation

- Segmentation nhạy cảm với biến đổi biên, texture và màu sắc.
- Augmentation giúp mô hình học được invariance với biến đổi hình học, noise và color shift.
- Kiểm thử độ bền vững của mask prediction khi dữ liệu bị degradation.

### Vai trò trong robustness analysis và corruption benchmark

- Robutness analysis đo lường sự suy giảm chất lượng khi dữ liệu bị corruption.
- Corruption benchmark giúp so sánh khả năng chịu nhiễu của các mô hình.
- Pipeline tạo dữ liệu theo mức độ (severity) để đo đường cong hiệu năng.

### Liên quan tới FastSAM và MobileSAM

- FastSAM và MobileSAM cần độ ổn định cao khi inference trên ảnh chất lượng thấp.
- Pipeline mô phỏng điều kiện thực tế (JPEG compression, noise, rotation, shear) để đánh giá segmentation fidelity.

---

## 2. Kiến trúc hệ thống

### Trách nhiệm của class `BasicAugmentation`

- Định nghĩa cấu hình augmentation.
- Quản lý registry của các transform.
- Thực thi augmentations theo batch từ thư mục input.
- Lưu ảnh kết quả và metadata để truy xuất.

### Workflow tổng thể

1. Đọc cấu hình và thiết lập RNG.
2. Đăng ký các transforms vào dictionary.
3. Duyệt toàn bộ ảnh trong `input_dir`.
4. Áp dụng từng transform và lưu ảnh.
5. Ghi metadata cho từng ảnh augmented.
6. (Tuỳ chọn) áp dụng hai transform ngẫu nhiên liên tiếp.
7. Xuất `metadata.json`.

### Transform registry

- Các transforms được lưu trong dictionary `self.transforms`.
- Key là tên transform, value là hàm thực thi.
- Hỗ trợ thêm mới bằng cách cập nhật dictionary.

### Metadata system

- Metadata lưu dạng list dictionary.
- Mỗi entry lưu `img_id`, `transform`, `save_path`.
- Xuất ra `metadata.json` để tái lập thí nghiệm.

### Image processing flow

- PIL đọc ảnh, resize về kích thước chuẩn.
- Albumentations xử lý các phép biến đổi.
- OpenCV xử lý color space.
- NumPy làm chuẩn dữ liệu và chuyển đổi giữa PIL/Albumentations.

### Dataset iteration flow

- Quét thư mục input.
- Lọc theo extension hợp lệ.
- Mỗi ảnh sinh ra 1 ảnh cho mỗi transform.

### Lý do thiết kế modular

- Dễ mở rộng bằng cách thêm transform mới.
- Cho phép bật/tắt hoặc thay đổi cấu hình dễ dàng.
- Tách riêng logic đọc/ghi ảnh, augmentation và metadata.

---

## 3. Giải thích Constructor (`__init__`)

### Tham số và cấu hình

| Tham số      | Kiểu dữ liệu          | Ý nghĩa                   | Ảnh hưởng đầu ra             | Ảnh hưởng robustness                       |
| ------------ | --------------------- | ------------------------- | ---------------------------- | ------------------------------------------ |
| `input_dir`  | `str`                 | Thư mục ảnh gốc           | Xác định tập nguồn           | Quyết định phân phối dữ liệu gốc           |
| `output_dir` | `str`                 | Thư mục lưu ảnh augmented | Tổ chức dữ liệu đầu ra       | Cho phép benchmark phân lớp theo transform |
| `size`       | `tuple(int,int)`      | Kích thước ảnh chuẩn      | Resize toàn bộ ảnh           | Chuẩn hoá input cho segmentation           |
| `seed`       | `int`                 | Seed random               | Tái lập kết quả              | Reproducibility trong robustness analysis  |
| `rng`        | `np.random.Generator` | Bộ sinh random            | Điều khiển sampling          | Đảm bảo consistency trong benchmark        |
| `config`     | `dict`                | Cấu hình hyperparameters  | Điều chỉnh mức độ corruption | Điều khiển độ khó của thử nghiệm           |
| `transforms` | `dict`                | Registry các augmentation | Quyết định loại biến đổi     | Đa dạng hóa phân phối dữ liệu              |
| `metadata`   | `list`                | Lưu log kết quả           | Tracking và audit            | Reproducibility                            |

### Ý nghĩa của từng thành phần

- `seed` và `rng` bảo đảm rằng cùng input sẽ tái sinh output tương tự, rất quan trọng trong nghiên cứu.
- `config` là trung tâm điều khiển severity, scale và mức độ nhiễu.
- `transforms` giúp pipeline mở rộng mà không thay đổi logic lõi.

---

## 4. Giải thích hệ thống config

| Field          | Kiểu dữ liệu  | Mục đích          | Cách hoạt động                     | Ảnh hưởng augmentation      | Ảnh hưởng segmentation      |
| -------------- | ------------- | ----------------- | ---------------------------------- | --------------------------- | --------------------------- |
| `rotate`       | `list[int]`   | Góc xoay          | Chọn ngẫu nhiên góc từ danh sách   | Tạo biến đổi orientation    | Ảnh hưởng biên và alignment |
| `severities`   | `list[int]`   | Mức độ corruption | Sample 1-3                         | Điều chỉnh noise/JPEG       | Kiểm tra độ bền mô hình     |
| `crop_resize`  | `list[float]` | Tỉ lệ crop        | RandomResizedCrop scale            | Mô phỏng zoom in/out        | Ảnh hưởng shape và context  |
| `shear`        | `list[int]`   | Góc shear         | Chọn ngẫu nhiên                    | Biến dạng hình học          | Biến dạng biên phân đoạn    |
| `color_jitter` | `dict`        | Biến đổi màu      | Brightness/contrast/saturation/hue | Mô phỏng ánh sáng, cảm biến | Ảnh hưởng texture và màu    |

---

## 5. Giải thích Utility Functions

### `_to_numpy`

- **Mục đích:** Chuẩn hoá ảnh về `np.ndarray`.
- **Input:** PIL Image hoặc ndarray.
- **Output:** ndarray.
- **Ý nghĩa:** Albumentations và OpenCV yêu cầu ndarray.

### `_read_image_path`

- **Mục đích:** Đọc và resize ảnh.
- **Input:** đường dẫn ảnh.
- **Output:** PIL Image đã resize.
- **Ý nghĩa:** Chuẩn hoá kích thước đầu vào.

### `_save_image`

- **Mục đích:** Lưu ảnh augmented.
- **Input:** PIL Image, tên file, subdir.
- **Output:** đường dẫn ảnh đã lưu.
- **Ý nghĩa:** Tổ chức output theo từng loại transform.

### `_apply_transform`

- **Mục đích:** Điều phối việc gọi transform.
- **Input:** ảnh, tên transform, hàm transform.
- **Output:** ảnh sau transform.
- **Ý nghĩa:** Inject logic severity cho gaussian/jepg.

### `_write_metadata_json`

- **Mục đích:** Xuất metadata ra JSON.
- **Input:** list metadata.
- **Output:** đường dẫn metadata.json.
- **Ý nghĩa:** Theo dõi thí nghiệm, reproducibility.

---

## 6. Giải thích chi tiết từng hàm augmentation

### 6.1 `_gaussian_noise`

**Ý nghĩa lý thuyết:** Gaussian noise mô hình hóa nhiễu cảm biến với phân phối chuẩn $\mathcal{N}(0, \sigma^2)$. Đây là corruption phổ biến trong môi trường ánh sáng kém hoặc cảm biến chất lượng thấp.

**Trực giác computer vision:** Nhiễu làm mờ ranh giới biên, tăng entropy của pixel và gây khó khăn cho segmentation khi mask dựa vào texture.

**Workflow:**

1. Chọn severity (1-3) từ `self.config`.
2. Mapping severity -> `std_range`.
3. Albumentations `A.GaussNoise` áp dụng noise.
4. Trả về ảnh nhiễu dạng ndarray.

**Hyperparameters và ảnh hưởng:**

- `std_range` lớn hơn làm noise mạnh hơn, giảm PSNR.
- Severity cao gây suy giảm mạnh ở vùng biên.

**Ảnh hưởng tới segmentation:**

- Boundary mờ, mask dễ bị răng cưa.
- Mô hình dễ bỏ sót vật thể nhỏ.

**Artifacts:** hạt noise lốm đốm, mất chi tiết mịn.

**Use cases:** mô phỏng ảnh công nghiệp noise cao, camera giá rẻ.

---

### 6.2 `_jpeg_noise`

**Ý nghĩa lý thuyết:** JPEG compression giảm dung lượng bằng cách lượng tử hóa DCT, gây mất thông tin ở tần số cao.

**Workflow:**

1. Mapping severity -> quality (25, 18, 12).
2. Convert ảnh sang uint8.
3. Lưu vào buffer JPEG với quality giảm dần.
4. Đọc lại ảnh từ buffer.

**Ảnh hưởng:**

- Block artifact, ringing, loss of texture.
- Segmentation bị ảnh hưởng ở biên mảnh.

**Artifacts:** block 8x8, màu bị banding.

**Use cases:** mô phỏng ảnh nén truyền tải qua mạng.

---

### 6.3 `_color_jitter`

**Ý nghĩa:** Random điều chỉnh brightness, contrast, saturation, hue.

**Trực giác:** Mô phỏng điều kiện ánh sáng, camera khác nhau.

**Workflow:**

1. Convert ảnh sang ndarray.
2. Khởi tạo `A.ColorJitter` với config.
3. Áp dụng biến đổi và trả về ảnh.

**Ảnh hưởng:**

- Mô hình học được invariance với ánh sáng.
- Có thể làm lệch histogram màu dẫn đến sai lệch segmentation nếu màu là cue chính.

---

### 6.4 `_random_rotation`

**Ý nghĩa:** Xoay ảnh nhằm mô phỏng orientation thay đổi.

**Workflow:**

1. Chọn góc từ `config.rotate`.
2. `A.Rotate` xoay ảnh theo góc đó.
3. Trả về ảnh mới.

**Ảnh hưởng:**

- Tăng khả năng nhận dạng vật thể không chuẩn orientation.
- Có thể tạo padding hoặc mất một phần biên.

---

### 6.5 `_flip`

**Ý nghĩa:** Horizontal/Vertical flip mô phỏng thay đổi góc nhìn.

**Workflow:**

1. RNG chọn ngang hoặc dọc.
2. Albumentations `HorizontalFlip` hoặc `VerticalFlip`.

**Ảnh hưởng:**

- Tăng tính đối xứng của dataset.
- Có thể gây phi thực tế nếu dữ liệu có orientation cố định.

---

### 6.6 `_translate`

**Ý nghĩa:** Dịch ảnh trong không gian (translation), mô phỏng camera shift.

**Workflow:**

1. RNG chọn shift_x, shift_y.
2. `A.Affine` với translate_percent.

**Ảnh hưởng:**

- Thay đổi vị trí vật thể.
- Tạo vùng trống ở biên, cần padding.

---

### 6.7 `_shear`

**Ý nghĩa:** Shear biến dạng hình học, mô phỏng góc nhìn xiên.

**Workflow:**

1. Chọn góc shear từ config.
2. `A.Affine` áp dụng shear.

**Ảnh hưởng:**

- Biến dạng biên, làm segmentation khó hơn.
- Hữu ích cho robustness đối với perspective shift.

---

### 6.8 `_cropping`

**Ý nghĩa:** Random crop + resize mô phỏng zoom và cắt khung hình.

**Workflow:**

1. Chọn scale từ config.
2. `A.RandomResizedCrop` crop rồi resize về `size`.

**Ảnh hưởng:**

- Thay đổi tỉ lệ vật thể trong ảnh.
- Có thể cắt mất một phần object gây khó khăn cho segmentation.

---

### 6.9 `_color_space`

**Ý nghĩa:** Chuyển đổi color space (HSV, LAB, GRAY).

**Workflow:**

1. Chọn mode ngẫu nhiên nếu chưa 지정.
2. OpenCV `cvtColor` đổi color space.
3. Gray được stack 3 kênh để giữ shape.

**Ảnh hưởng:**

- Mô phỏng cảm biến khác nhau.
- Có thể làm mô hình robust hơn với biến đổi color domain.

---

### 6.10 `_random_two_augmentations`

**Ý nghĩa:** Kết hợp hai augmentation liên tiếp để tạo corruption phức hợp.

**Workflow:**

1. Chọn ngẫu nhiên 2 transform khác nhau từ registry.
2. Áp dụng transform thứ nhất.
3. Áp dụng transform thứ hai.
4. Trả về ảnh và tên transform.

**Ảnh hưởng:**

- Tạo distribution phức tạp hơn, gần thực tế.
- Tăng nguy cơ artifacts và out-of-distribution.

---

### Albumentations, OpenCV, PIL và randomness

- Albumentations xử lý core augmentation với API `transform(image=...)` trả về dict.
- OpenCV dùng cho chuyển đổi color space chính xác và nhanh.
- PIL dùng cho đọc ảnh, resize và lưu JPEG.
- Randomness điều khiển bởi `np.random.default_rng(seed)`.

---

## 7. Hệ thống Severity-Based Corruption

### Khái niệm severity

- Severity là thước đo mức độ corruption.
- Severity càng cao, chất lượng ảnh càng giảm.
- Quan trọng trong benchmark để vẽ curve hiệu năng theo mức độ nhiễu.

### Sampling severity

- Severity được chọn ngẫu nhiên từ `config.severities`.
- Gaussian và JPEG sử dụng severity để map sang tham số cụ thể.

### Mapping

| Severity | Gaussian `std_range` | JPEG Quality |
| -------- | -------------------- | ------------ |
| 1        | 0.05 - 0.10          | 25           |
| 2        | 0.10 - 0.18          | 18           |
| 3        | 0.18 - 0.30          | 12           |

---

## 8. Workflow của `apply_augmentation()`

1. **Scan directory:** đọc danh sách file trong `input_dir`.
2. **Validate extension:** lọc theo `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.
3. **Load image:** dùng PIL mở ảnh.
4. **Resize image:** resize về `self.size`.
5. **Iterate transforms:** duyệt từng transform trong registry.
6. **Apply augmentation:** gọi `_apply_transform`.
7. **Convert format:** đảm bảo ảnh là uint8, chuyển về PIL nếu cần.
8. **Save image:** lưu theo subdir tên transform.
9. **Generate metadata:** append vào list.
10. **Random two augmentations:** nếu bật, áp dụng 2 transform liên tiếp.
11. **Export metadata.json:** ghi toàn bộ log vào file.

---

## 9. Random Two-Augmentation Strategy

- **Lý do kết hợp:** corruption thực tế thường xảy ra đồng thời (ví dụ noise + JPEG).
- **Sampling without replacement:** đảm bảo hai transform khác nhau để tăng đa dạng.
- **Lợi ích:** tăng độ phong phú của distribution, đánh giá robust hơn.
- **Rủi ro:** có thể tạo artifacts không thực tế nếu quá mạnh.

---

## 10. Metadata System

### Cấu trúc metadata

```json
{
  "img_id": "0001",
  "transform": "gaussian",
  "save_path": ".../gaussian/0001_gaussian.png"
}
```

### Vai trò

- Theo dõi dữ liệu đã sinh.
- Reproducibility trong benchmark.
- Audit và experiment tracking.

---

## 11. Resize Strategy

- Resize toàn bộ ảnh về `size` giúp mô hình nhận input đồng nhất.
- PIL resize sử dụng resampling mặc định (thường là BICUBIC cho ảnh RGB).
- Downsampling có thể mất chi tiết biên, ảnh hưởng segmentation.
- Nhưng tăng tính nhất quán và giúp so sánh công bằng giữa các mô hình.

---

## 12. Libraries và Dependencies

| Thư viện       | Vai trò                                 |
| -------------- | --------------------------------------- |
| OpenCV         | Color space conversion, xử lý ảnh nhanh |
| PIL            | Đọc, resize, lưu ảnh                    |
| NumPy          | Chuẩn hoá data và chuyển đổi kiểu       |
| Albumentations | Core augmentation pipeline              |
| tqdm           | Progress bar khi xử lý batch            |
| json           | Lưu metadata                            |

---

## 13. Góc nhìn Research và Robustness

- Pipeline tạo benchmark có kiểm soát để đánh giá robustness.
- Hỗ trợ ablation study bằng cách tắt/bật từng transform.
- Có thể đo độ nhạy của model với từng loại corruption.
- Phù hợp cho các nghiên cứu về segmentation stability.

---

## 14. Advantages và Limitations

### Strengths

- Modular, dễ mở rộng.
- Có metadata tracking.
- Severity-based corruption rõ ràng.

### Weaknesses

- Xử lý tuần tự, không tối ưu tốc độ.
- Không hỗ trợ mask augmentation đồng bộ (segmentation ground truth).
- Resize cố định có thể làm mất chi tiết.

### Scalability concerns

- Dataset lớn sẽ tốn I/O và thời gian.
- Không sử dụng multiprocessing hoặc GPU.

### Possible failure cases

- Augmentation quá mạnh gây mất object.
- JPEG noise ở quality thấp có thể làm mask không chính xác.

---

## 15. Future Improvements

- Bổ sung augmentation đồng bộ cho mask.
- Hỗ trợ multiprocessing hoặc GPU acceleration.
- Cho phép custom pipeline bằng YAML config.
- Thêm logging nâng cao (MLflow, Weights&Biases).
- Hỗ trợ online augmentation trong training loop.

---

## 16. Conclusion

Pipeline augmentation này cung cấp một nền tảng chuẩn hóa để đánh giá robustness cho segmentation. Thiết kế modular giúp mở rộng dễ dàng, metadata giúp tái lập thí nghiệm, và severity-based corruption hỗ trợ benchmark có hệ thống. Đây là nền tảng hữu ích cho cả nghiên cứu học thuật và hệ thống ML production.

---

## Ví dụ sử dụng

```python
from augmentations import BasicAugmentation

aug = BasicAugmentation(
    input_dir="raw_data/hazelnut/train/good",
    output_dir="processed_data/version_3",
    size=(512, 512),
)

aug.apply_augmentation(include_random_two=True, random_two_count=2)
```
