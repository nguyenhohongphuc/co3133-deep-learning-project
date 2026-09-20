# Hướng dẫn thực hiện task CO3133 — Bài tập lớn 1 & Bài tập lớn 2

Tài liệu tổng hợp yêu cầu của handbook cho từng task trong bảng phân công của nhóm (`CO3133_Phan_Cong_A1_A2.xlsx`).

- **Nguồn:** `handbook-ene-v2.pdf`, bản sửa đổi ngày 14/09/2026. Ký hiệu `§` là số mục trong handbook.
- **Thứ tự:** các task được xếp theo đúng thứ tự thời gian như trong bảng (STT 1–31).
- **Quy ước:** mọi nội dung dưới tiêu đề *Theo handbook* đều lấy từ file PDF. Những dòng ghi **Kế hoạch nhóm** là lựa chọn của nhóm, không phải yêu cầu của handbook.
- **Người phụ trách và trạng thái** được cập nhật trong bảng tính; tài liệu này dùng để tra cứu *thế nào thì được coi là xong*.

---

## Mục lục

| STT | Mã | Task | Phụ trách | Ưu tiên | DL nội bộ | DL chính thức |
|---|---|---|---|---|---|---|
| — | — | Quy tắc chung cho mọi task | — | — | — | — |
| 1 | A1-01 | Data pipeline | TV A | P1 | 21/09 | 23/09 |
| 2 | A1-02 | EDA Fashion-MNIST | TV A | P1 | 21/09 | 23/09 |
| 3 | A1-03 | Linear + MLP | TV A | P1 | 22/09 | 23/09 |
| 4 | A1-04 | Nộp A1 M1 Draft | TV A | **P0** | 22/09 | 23/09 |
| 5 | A2-01 | Chọn task & dataset A2 | TV A | P1 | 24/09 | 26/09 |
| 6 | A1-05 | CNN + LSTM/GRU | TV B | P1 | 29/09 | 01/10 |
| 7 | A1-06 | Transformer | TV C | P1 | 29/09 | 01/10 |
| 8 | A2-02 | Nộp A2 M1 Proposal | TV A | **P0** | 05/10 | 07/10 |
| 9 | A1-07 | Multi-seed & Ablation | TV B | P1 | 12/10 | 14/10 |
| 10 | A1-08 | Phân tích lỗi & so sánh | TV C | P1 | 12/10 | 14/10 |
| 11 | A2-03 | Pipeline & EDA sơ bộ A2 | TV A | P2 | 12/10 | 14/10 |
| 12 | A1-09 | Báo cáo A1 | Cả nhóm | P2 | 19/10 | 21/10 |
| 13 | A1-10 | Slide A1 | TV B | P2 | 19/10 | 21/10 |
| 14 | A1-11 | Video A1 | Cả nhóm | P2 | 20/10 | 21/10 |
| 15 | A1-12 | Web A1 & AI disclosure | TV C | P2 | 20/10 | 21/10 |
| 16 | A1-13 | Checkpoints & README A1 | TV A | P2 | 19/10 | 21/10 |
| 17 | A1-14 | Nộp A1 M2 Final | Cả nhóm | **P0** | 20/10 | 21/10 |
| 18 | A2-04 | Cổng phê duyệt Proposal A2 | TV A | **P0** | 19/10 | 21/10* |
| 19 | A2-05 | Baseline A2 | TV A | P1 | 26/10 | 28/10 |
| 20 | A2-06 | Fine-tune pretrained | TV B | P1 | 26/10 | 28/10 |
| 21 | A2-07 | Metrics & web nháp A2 | TV C | P1 | 26/10 | 28/10 |
| 22 | A2-08 | Nộp A2 M2 Draft | Cả nhóm | **P0** | 27/10 | 28/10 |
| 23 | A2-09 | Ablation A2 | TV B | P1 | 02/11 | 04/11 |
| 24 | A2-10 | Định tính & phân tích lỗi A2 | TV C | P1 | 02/11 | 04/11 |
| 25 | A2-11 | Compute cost A2 | TV A | P1 | 09/11 | 11/11 |
| 26 | A2-12 | Báo cáo A2 | Cả nhóm | P2 | 09/11 | 11/11 |
| 27 | A2-13 | Slide A2 | TV C | P2 | 09/11 | 11/11 |
| 28 | A2-14 | Video A2 | Cả nhóm | P2 | 10/11 | 11/11 |
| 29 | A2-15 | Web A2 & AI disclosure | TV A | P2 | 10/11 | 11/11 |
| 30 | A2-16 | Checkpoints & README A2 | TV B | P2 | 09/11 | 11/11 |
| 31 | A2-17 | Nộp A2 M3 Final | Cả nhóm | **P0** | 10/11 | 11/11 |

Tất cả các ngày đều thuộc năm 2026. \* Handbook không nêu ngày giảng viên duyệt Proposal; 21/10 là hạn do nhóm tự đặt (xem A2-04).

---

## Quy tắc chung cho mọi task

**Deadline và nộp trễ**
- Mọi cột mốc đều hết hạn lúc **23:59 giờ Việt Nam (GMT+7)** của ngày tương ứng, trừ khi LMS thông báo khác cho cột mốc đó (§1.4, §7.1).
- Bài nộp trễ được chấp nhận tối đa **một tuần**. Mức phạt là **20% điểm của cột mốc đó cho mỗi tuần trễ, tính cả phần lẻ của tuần**, và chỉ áp dụng cho cột mốc đó (§7.2).
  - Ví dụ trong handbook: bản M1 Draft được 80/100 nhưng nộp trễ 2 ngày thì chỉ còn 0,80 × 80 = 64 điểm (§7.2).
- Trễ quá 7 ngày thì cột mốc đó bị **0 điểm**, trừ khi giảng viên đã đồng ý ngoại lệ bằng văn bản từ trước (§7.2).
- Nếu thông báo trên LMS cho một cột mốc khác với handbook, hãy làm theo thông báo LMS cho cột mốc đó (§7.8).

**Cách tính điểm** (§1.3, §6.1)
- Điểm đồ án = A1 **40%** + A2 **30%** + A3 **30%**.
- Mỗi bài được chấm trên thang 100 theo rubric riêng, sau đó nhân với trọng số. Ví dụ: A1 được 85 điểm → đóng góp 0,40 × 85 = 34 điểm.
- Trọng số các cột mốc trong từng bài:
  - A1: M1 Draft 25%, M2 Final 75%.
  - A2: M1 Proposal 15%, M2 Draft 25%, M3 Final 60%.

**Nguyên tắc chung** (§1.5, §6.2)
- Phân biệt rõ phần **bắt buộc** và phần **tùy chọn**.
- Không dùng nhận xét mơ hồ như "dataset đủ lớn", "kết quả tốt", "phân tích đầy đủ" mà không có tiêu chí định lượng.
- Chỉ dựa vào accuracy thì **không đủ** để kết luận phương pháp này tốt hơn phương pháp khác.
- Số liệu cao nhưng không giải thích được hoặc không tái lập được thì không đạt điểm tối đa.
- Phần bắt buộc chiếm phần lớn số điểm; phần mở rộng **không thể thay thế** phần bắt buộc còn thiếu.
- Tiêu chí chấm gồm: tính đúng đắn, mức độ hiểu bài, thiết kế thí nghiệm, khả năng tái lập, phân tích, chất lượng code, chất lượng báo cáo, trình bày và liêm chính học thuật.

**Sử dụng AI** (§5, §5.5)
- Chỉ được dùng công cụ AI khi **khai báo đầy đủ và kiểm chứng được**.
- Mỗi thành viên chịu trách nhiệm hoàn toàn về code, số liệu, hình, nhận định và nội dung đã nộp, và phải giải thích được mọi phần.
- Không nộp output AI chưa kiểm tra. Không dùng AI để bịa dữ liệu, kết quả, trích dẫn hay tài liệu tham khảo.
- **Không được khai báo những thí nghiệm không thực sự chạy.**
- Không dán dữ liệu riêng tư, bị hạn chế hoặc chứa thông tin đăng nhập vào công cụ AI.
- Khai báo thiếu hoặc sai có thể bị xem là vi phạm liêm chính học thuật.
- Giảng viên có thể phỏng vấn bất kỳ thành viên nào, yêu cầu giải thích, sửa code trực tiếp, diễn giải metric hoặc chạy lại một phần. Mức đóng góp cá nhân có thể ảnh hưởng đến điểm cuối cùng.

**Khả năng tái lập** (§4.2)
- Mọi kết quả chính phải truy được về **cấu hình mô hình, cách chia dữ liệu, checkpoint, log/mã thí nghiệm và commit/tag** tương ứng.
- Chỉ nộp một notebook đã chạy sẵn mà không có hướng dẫn tái lập là không được chấp nhận.

---

# Giai đoạn 1 — A1 M1 Draft (hạn 23/09/2026)

### A1-01 · Data pipeline

> **TV A** · P1 · DL nội bộ 21/09 · DL chính thức 23/09
> **Kết quả cần đạt:** split train/val/test cố định (seed + số mẫu) + DataLoader + vòng lặp train/val có log

**Theo handbook**
- **Vai trò của từng dataset** (§10):
  - **MNIST** chỉ được dùng để phát triển và debug.
  - **Kết quả chính phải báo cáo trên Fashion-MNIST.**
  - CIFAR-10 là phần mở rộng tùy chọn.
- **Một split cho mọi mô hình** (§10, §12.2): tất cả mô hình trong phần so sánh chính phải dùng **cùng các tập train/validation/test và cùng random seed**.
- **Báo cáo về split** (§10): split chính thức hay tự chia, seed, cách phân tầng (nếu có) và **số mẫu chính xác của từng tập**.
- **Mục chia dữ liệu trong báo cáo** (§3.1): train/val/test; split chính thức hay tự chia; seed; phân tầng; đơn vị chia; cách chống rò rỉ dữ liệu (leakage); quy tắc chọn tập con (nếu có).
- **Mục tiền xử lý trong báo cáo** (§3.1): làm sạch; resize/chuẩn hóa; augmentation; xử lý nhãn lỗi; **kích thước tensor đầu vào**; **một batch mẫu sau tiền xử lý**.
- **Phạm vi** (§8.1): thiết kế Dataset và DataLoader, tiền xử lý và augmentation, pipeline train/validation/test **bằng PyTorch**.
- **Framework** (§11.3): PyTorch là framework chính. Code phải là một project tái lập được, không phải một notebook chạy một lần không kèm hướng dẫn.
- **Ghi lại cấu hình huấn luyện** (§3.2): optimizer, learning rate, batch size, số epoch, scheduler, regularization, early stopping, **tiêu chí chọn checkpoint**, seed, phần cứng, mixed precision (nếu dùng), thời gian train, cách chọn siêu tham số.
- **Sơ đồ pipeline** bắt buộc có trong báo cáo (§3.2):
  `Raw data → preprocessing → data loader → model → loss → optimization → prediction → post-processing → evaluation`
- **Repository** (§4.2): phải có file cấu hình và thiết lập seed.

**Ảnh hưởng đến điểm** (§14)
- *Data & training pipeline* (15%): Dataset/DataLoader, tiền xử lý, vòng lặp train/val/test.
- *Problem & data understanding* (15%): gồm cả cách chia dữ liệu và **ý thức về leakage**.

**Kế hoạch nhóm**
- Giữ nguyên tập test chính thức 10k; chia tập train chính thức thành 54k/6k train/val có phân tầng; lưu lại chỉ số (indices) của split để mọi mô hình dùng chung.

---

### A1-02 · EDA Fashion-MNIST

> **TV A** · P1 · DL nội bộ 21/09 · DL chính thức 23/09
> **Kết quả cần đạt:** phân bố lớp; kích thước ảnh; mất cân bằng; ảnh mẫu

**Theo handbook**
- **EDA tối thiểu cho A1** (§13): **phân bố lớp, kích thước đầu vào, phân tích mất cân bằng, ảnh mẫu đại diện.**
- **Mô tả dataset trong báo cáo** (§3.1): nguồn, phiên bản, giấy phép; quy trình thu thập/gán nhãn nếu biết; số mẫu và nhãn; phân bố lớp; dung lượng lưu trữ và định dạng; **thiên lệch (bias) và hạn chế**.
- **Nêu rõ vai trò dataset** (§10): dataset nào dùng để debug, dataset nào dùng cho so sánh chính, dataset nào (nếu có) dùng cho phần mở rộng.
- **Trang assignment** (§2.2) phải có mục "Dataset description and EDA".
- **Không nhận xét mơ hồ** (§1.5): mọi nhận định phải có số liệu đi kèm.

**Lưu ý**
- Fashion-MNIST là dataset cân bằng, nhưng vẫn bắt buộc phải có phân tích mất cân bằng. Hãy đưa số mẫu từng lớp để chứng minh.

---

### A1-03 · Linear + MLP

> **TV A** · P1 · DL nội bộ 22/09 · DL chính thức 23/09
> **Kết quả cần đạt:** 2 mô hình train được trên Fashion-MNIST (không softmax trước CrossEntropyLoss)

**Theo handbook**
- **Bộ phân loại Linear / softmax** (§11.1):
  - Duỗi (flatten) mỗi ảnh thành một vector.
  - Dùng một lớp linear để sinh logits.
  - Dùng hàm mất mát cross-entropy.
  - **Không áp dụng softmax trước `CrossEntropyLoss`.**
- **MLP** (§11.1):
  - Có ít nhất một lớp ẩn.
  - Giải thích các hàm kích hoạt và các kỹ thuật regularization đã dùng.
- **Mô tả từng phương pháp trong báo cáo** (§3.2): kiến trúc, cách biểu diễn đầu vào, loss, metric, điểm mạnh và hạn chế, **số tham số**. Nêu rõ phần nào tự cài đặt, phần nào dùng thư viện (ghi tên thư viện).
- **Tự cài đặt và tự train** (§11.3): backbone pretrained chỉ được dùng như phần mở rộng hoặc mốc tham chiếu cho ablation.

**Ảnh hưởng đến điểm**
- Linear + MLP nằm trong **yêu cầu tối thiểu của M1** (§7.4).
- *Methods & implementation* là mục có trọng số lớn nhất của A1, **30%**: đủ 5 mô hình bắt buộc, loss đúng, có giải thích (§14).

---

### A1-04 · Nộp A1 M1 Draft

> **TV A** · **P0** · DL nội bộ 22/09 · DL chính thức **23/09/2026 23:59**
> **Kết quả cần đạt:** bản nháp + repo chạy được; có link trên landing page (25% điểm A1)

**Theo handbook**
- **Trọng số** (§7.4): M1 Draft = **25% điểm Bài tập lớn 1** (tương đương 10 điểm đồ án).
- **Yêu cầu tối thiểu** (§7.4):
  - EDA
  - Dataset/DataLoader
  - vòng lặp train/val
  - **Linear + MLP chạy được**
  - CNN là tùy chọn
- **Chưa bắt buộc** (§7.4): LSTM/GRU và Transformer bắt buộc ở M2 Final, không bắt buộc ở M1 Draft.
- **Thế nào là bản nháp** (§7.7): cột mốc Draft chấp nhận **báo cáo chưa hoàn chỉnh và repo đang làm dở**. Đây là mốc kiểm tra tiến độ có chấm điểm.
- **Nộp ở đâu** (§7.1): bản Draft có thể đặt link trên landing page. Quy định nộp PDF lên LMS chỉ áp dụng cho báo cáo **Final** của mỗi bài.
- **Landing page** (§1.2, §4.1): phải có link đến mọi sản phẩm bắt buộc ngay khi sản phẩm đó sẵn sàng.
- **Nộp trễ** (§7.2): xem mục *Quy tắc chung cho mọi task* ở đầu tài liệu.

**Kiểm tra trước khi nộp**
- [ ] Đã có hình/bảng EDA
- [ ] DataLoader + vòng lặp train/val chạy được
- [ ] Linear và MLP train được trên Fashion-MNIST
- [ ] Bản nháp có link trên landing page và các link mở được
- [ ] `AI_USAGE.md` đã ghi lại mọi lần dùng AI tính đến thời điểm này (§5.1)

---

# Giai đoạn 2 — Từ sau M1 Draft đến hạn Proposal A2 (24/09 – 07/10)

### A2-01 · Chọn task & dataset A2

> **TV A** (B & C duyệt) · P1 · DL nội bộ 24/09 · DL chính thức 26/09
> **Kết quả cần đạt:** chọn track + 2–3 dataset đạt ngưỡng §19 + ước tính GPU

**Theo handbook**
- **Các track** (§17), chọn **một**:
  - phân loại ảnh (image classification)
  - phân loại văn bản hoặc gán nhãn chuỗi (text classification / sequence labeling)
  - phát hiện đối tượng (object detection)
  - phân đoạn ngữ nghĩa hoặc phân đoạn thể hiện (semantic / instance segmentation)
  - nhận dạng lại người/phương tiện (person/vehicle re-identification)
  - ước lượng độ sâu từ một ảnh (monocular depth estimation)
  - một task chuyên biệt khác được giảng viên duyệt
- **Làm sâu hơn làm rộng** (§17): yêu cầu làm sâu một track, không cần làm nhiều track.
- **Ngưỡng dataset mặc định** (§19):

  | Track | Ngưỡng tối thiểu | Yêu cầu thêm |
  |---|---|---|
  | Phân loại ảnh | ≥5 lớp; ≥5.000 mẫu train; đủ mẫu mỗi lớp để đánh giá có ý nghĩa | **Không dùng MNIST hoặc Fashion-MNIST** |
  | Phân loại văn bản | ≥5 lớp; ≥5.000 mẫu train | Kiểm tra dữ liệu trùng lặp, phân bố độ dài văn bản, mất cân bằng lớp; mô tả cách tokenize và chuẩn hóa |
  | Object detection | ≥5 lớp; khoảng 3.000+ ảnh train; khoảng 5.000+ đối tượng | Phân tích phân bố kích thước bounding box và số đối tượng mỗi ảnh |
  | Segmentation | ≥3 lớp foreground (không tính background); khoảng 1.000+ ảnh có mask | Dataset nhị phân chỉ được dùng nếu thực sự khó và được duyệt; phân tích phân bố pixel theo lớp |
  | Re-identification | khoảng 300+ danh tính (identity) để train; nhiều ảnh mỗi danh tính; ưu tiên nhiều camera | **Danh tính ở tập test không được trùng với tập train**; dùng đúng giao thức query/gallery |
  | Ước lượng độ sâu | khoảng 5.000+ cặp RGB–depth | Mô tả thang đo độ sâu, pixel không hợp lệ, cách chia theo cảnh; **cùng một cảnh không được nằm ở cả train và test** |

- **Ngoại lệ** (§19): chỉ được chấp nhận khi được duyệt rõ ràng và có bằng chứng về độ khó của task, giá trị học thuật, hoặc ràng buộc đặc biệt về nhãn/tài nguyên tính toán.
- **Metric bắt buộc** (§20.1). Chọn dataset tính được các metric này:
  - Phân loại: accuracy + macro-F1
  - Detection: mAP + AP theo lớp và/hoặc theo kích thước
  - Segmentation: mIoU + Dice (+ mask AP nếu là instance segmentation)
  - Re-ID: Rank-1, Rank-5, mAP
  - Độ sâu: Abs Rel, RMSE, δ accuracy
  - Task văn bản: metric phù hợp **kèm lý giải rõ ràng**
- **Cổng phê duyệt** (§18): dataset chỉ được dùng cho phần chính của bài **sau khi được duyệt**.

**Kế hoạch nhóm**
- Ước tính GPU ở bước này sẽ dùng cho mục *compute estimate* bắt buộc trong Proposal (§18).

---

### A1-05 · CNN + LSTM/GRU

> **TV B** · P1 · DL nội bộ 29/09 · DL chính thức 01/10
> **Kết quả cần đạt:** CNN tự thiết kế + RNN trên chuỗi hàng/patch; cả hai train được trên pipeline chung

**Theo handbook**
- **CNN** (§11.1):
  - **Tự thiết kế kiến trúc.**
  - **Không** được chỉ gọi một mô hình pretrained rồi nộp làm CNN.
  - Giải thích convolution, pooling và feature map.
- **LSTM hoặc GRU** (§11.1):
  - Biểu diễn mỗi ảnh thành một chuỗi các **hàng, cột hoặc patch**.
  - Giải thích **cách định nghĩa timestep, kích thước đầu vào và biểu diễn ẩn (hidden representation)**.
- **Mở rộng tùy chọn** (§11.2): so sánh LSTM với GRU. Phần mở rộng không thay thế được mô hình bắt buộc.
- **Công bằng** (§12.2): dùng cùng split, cùng giao thức đánh giá và metric với các mô hình chính khác.
- **Mô tả từng phương pháp trong báo cáo** (§3.2): kiến trúc, biểu diễn đầu vào, loss, metric, điểm mạnh và hạn chế, **số tham số**, phần tự cài đặt và phần dùng thư viện.
- **Mục tiêu học tập** (§9): phân tích inductive bias và cách biểu diễn dữ liệu (pixel được duỗi phẳng, đặc trưng không gian, chuỗi hàng/cột/patch).

**Ảnh hưởng đến điểm** (§14)
- *Methods & implementation* (30%): đủ 5 mô hình bắt buộc, loss đúng, có giải thích.

---

### A1-06 · Transformer

> **TV C** · P1 · DL nội bộ 29/09 · DL chính thức 01/10
> **Kết quả cần đạt:** patch embedding + positional encoding, train được; A1 đã đủ 5 mô hình

**Theo handbook**
- **Transformer** (§11.1):
  - Biểu diễn mỗi ảnh thành các hàng/cột hoặc patch.
  - Có **token embedding/projection** và **positional encoding**.
  - **Giải thích đầu vào và đầu ra của attention.**
- **Tùy chọn** (§11.2): Mamba hoặc các state-space model khác **không bắt buộc**.
- **Tự cài đặt và tự train** (§11.3): backbone pretrained không được tính là mô hình bắt buộc.
- **Mô tả phương pháp trong báo cáo** (§3.2): kiến trúc, biểu diễn đầu vào, loss, metric, điểm mạnh và hạn chế, số tham số, phần tự cài đặt và phần dùng thư viện.
- **Công bằng** (§12.2): cùng split, giao thức đánh giá và metric với các mô hình khác.

**Lưu ý**
- Sau task này phải có đủ 5 mô hình bắt buộc: Linear, MLP, CNN, LSTM/GRU, Transformer (§11.1). Cả 5 đều bắt buộc ở A1 Final (§7.4).

---

### A2-02 · Nộp A2 M1 Proposal

> **TV A** (B & C review bản ngày 05/10) · **P0** · DL nội bộ 05/10 · DL chính thức **07/10/2026 23:59**
> **Kết quả cần đạt:** đủ 11 mục theo §18

**Theo handbook**
- **Trọng số** (§7.5): M1 Proposal = **15% điểm Bài tập lớn 2**.
- **Nội dung bắt buộc** (§18):
  1. Tên dataset, nguồn, phiên bản và giấy phép
  2. Định nghĩa task, đầu vào và đầu ra
  3. Số mẫu và loại nhãn
  4. Phân tích phân bố sơ bộ
  5. Kế hoạch chia dữ liệu
  6. **Đơn vị chia dữ liệu để chống leakage**
  7. Metric đánh giá
  8. Kế hoạch baseline
  9. Mô hình pretrained dự kiến
  10. **Ước tính tài nguyên tính toán**
  11. Quy tắc chọn tập con, nếu có
- **Các trạng thái duyệt** (§18): Approved (Đã duyệt); Approved with conditions (Duyệt có điều kiện); Rejected (Từ chối).
- **Cổng phê duyệt** (§7.1, §7.5, §15.1): chỉ được bắt đầu thí nghiệm chính **sau khi được Approved hoặc Approved with conditions**. Phải chờ duyệt trước khi triển khai đầy đủ.
- **Proposal nộp trễ** (§7.2): vẫn phải được duyệt trước khi làm phần chính, và vẫn bị trừ điểm trễ ở cột mốc M1.
- **Lưu hồ sơ** (§4.1): hồ sơ Dataset Proposal và trạng thái duyệt là sản phẩm bắt buộc của A2.

**Ảnh hưởng đến điểm** (§23)
- *Problem & data understanding* (15%): **chất lượng proposal**, EDA, kiểm soát split/leakage.

---

# Giai đoạn 3 — Thí nghiệm A1 và chuẩn bị A2 (08/10 – 14/10)

### A1-07 · Multi-seed & Ablation

> **TV B** · P1 · DL nội bộ 12/10 · DL chính thức 14/10
> **Kết quả cần đạt:** 5 mô hình × 3 seed (mean ± std) + ≥1 ablation

**Theo handbook**
- **Kết quả định lượng** (§3.3): metric chính và phụ, số tham số, thời gian train/inference, cấu hình, và **mean ± độ lệch chuẩn qua nhiều lần chạy khi khả thi**.
- **Ablation** (§3.3): bắt buộc ít nhất một ablation cho A2 và A3; với A1 thì **khuyến khích khi so sánh các cách biểu diễn hoặc các kỹ thuật regularization**.
- **Thiết kế thí nghiệm, cho từng thí nghiệm** (§3.2): **giả thuyết; yếu tố thay đổi; các yếu tố giữ cố định; metric; tiêu chí kết luận.**
- **Công bằng** (§12.2): cùng split cho mọi mô hình chính; cùng giao thức đánh giá và metric; báo cáo phần cứng, phiên bản phần mềm, seed và quy tắc chọn checkpoint.
- **Truy vết** (§4.2): mỗi kết quả phải gắn với config, split, checkpoint, log/mã thí nghiệm và commit/tag.
- **Liêm chính** (§5.5): không khai báo những thí nghiệm không thực sự chạy.

**Kế hoạch nhóm**
- Dùng 3 seed và chọn "hàng vs patch" làm ablation là lựa chọn của nhóm, không phải yêu cầu của handbook.

---

### A1-08 · Phân tích lỗi & so sánh

> **TV C** · P1 · DL nội bộ 12/10 · DL chính thức 14/10
> **Kết quả cần đạt:** bảng Acc/Macro-F1/params/thời gian; confusion matrix; ví dụ dự đoán sai; phân tích inductive bias

**Theo handbook**
- **Các đầu ra so sánh bắt buộc** (§12.1):
  - Accuracy
  - Macro-F1
  - Số tham số
  - Thời gian train
  - Thời gian inference
  - Đường cong training và validation
  - Confusion matrix
  - Ví dụ dự đoán đúng và sai
  - Phân tích cách biểu diễn dữ liệu và inductive bias ảnh hưởng đến hiệu năng
- **Không kết luận mô hình nào tốt hơn chỉ dựa vào accuracy** (§12.1).
- **Phần so sánh và thảo luận phải trả lời** (§3.3):
  - phương pháp nào tốt hơn và **tốt hơn bao nhiêu**
  - khác biệt xuất hiện ở đâu
  - nguyên nhân liên quan đến dữ liệu, kiến trúc, loss hay quá trình train
  - đánh đổi giữa accuracy – tốc độ – số tham số
  - giả thuyết có được ủng hộ hay không
  - những kết luận **chưa thể** rút ra
  - Chỉ liệt kê metric rồi chọn điểm cao nhất là "tốt nhất" thì **không đạt yêu cầu**.
- **Phân tích lỗi, cho từng nhóm lỗi** (§3.3): mô tả; tần suất/tỷ lệ nếu có thể; ví dụ; nguyên nhân giả định; hướng cải thiện.
- **Kết quả định tính** (§3.3): dự đoán đúng; trường hợp khó nhưng vẫn đúng; trường hợp sai; so sánh dự đoán với nhãn thật.
- **Hành vi huấn luyện** (§3.3): đường cong loss/metric; checkpoint được chọn; overfitting/underfitting; thời gian train; các vấn đề khi cài đặt và cách khắc phục.

**Ảnh hưởng đến điểm** (§14)
- *Experimental design & results* (20%): so sánh công bằng, metric ngoài accuracy, đường cong/bảng.
- *Analysis & error analysis* (10%): thảo luận inductive bias, trường hợp sai, hạn chế.

---

### A2-03 · Pipeline & EDA sơ bộ A2

> **TV A** · P2 · DL nội bộ 12/10 · DL chính thức 14/10
> **Kết quả cần đạt:** DataLoader cho toàn bộ dataset + EDA theo track; chưa train chính

**Theo handbook**
- **Cổng phê duyệt** (§7.5, §15.1): thí nghiệm chính chỉ bắt đầu sau khi được duyệt; phải chờ duyệt trước khi triển khai đầy đủ. Task này chỉ mang tính chuẩn bị, sẵn sàng điều chỉnh nếu proposal được duyệt kèm điều kiện.
- **EDA theo từng track** (§22):
  - Phân loại: phân bố lớp, kích thước/độ dài đầu vào, mất cân bằng
  - Detection: số lượng đối tượng, phân bố lớp, kích thước bounding box
  - Segmentation: phân bố pixel, kích thước vùng, chất lượng mask
  - Re-ID: phân bố danh tính, số ảnh mỗi danh tính, phân bố camera
  - Độ sâu: phân bố độ sâu, pixel không hợp lệ, phân bố cảnh
- **Kiểm tra dữ liệu theo track** (§19): xem bảng ở task A2-01 (trùng lặp và tokenize với văn bản; kích thước box với detection; pixel theo lớp với segmentation; chia theo danh tính với Re-ID; thang độ sâu và chia theo cảnh với depth).
- **Chia dữ liệu và tiền xử lý** (§3.1): yêu cầu báo cáo giống A1-01, bao gồm **đơn vị chia và cách chống leakage**.

**Ảnh hưởng đến điểm** (§23)
- *Data & training pipeline* (15%): **nạp dữ liệu quy mô lớn**, tiền xử lý, vòng lặp train.

---

# Giai đoạn 4 — Gói nộp A1 Final (hạn 21/10/2026)

### A1-09 · Báo cáo A1

> **Cả nhóm** (mỗi người viết phần mô hình mình làm) · P2 · DL nội bộ 19/10 · DL chính thức 21/10
> **Kết quả cần đạt:** file PDF đủ 3 phần theo §3

**Theo handbook — cấu trúc bắt buộc** (§3)
- **Phần 1 — Bài toán và dữ liệu** (§3.1)
  - *Phát biểu bài toán:* động lực thực tế; đầu vào; đầu ra; đơn vị dự đoán; công thức toán học khi phù hợp; các khó khăn chính.
  - *Mô tả dataset:* nguồn, phiên bản, giấy phép; quy trình thu thập/gán nhãn; số mẫu; phân bố lớp; dung lượng và định dạng; thiên lệch và hạn chế.
  - *EDA:* với A1 tối thiểu gồm phân bố lớp, kích thước đầu vào, phân tích mất cân bằng, ảnh mẫu đại diện (§13).
  - *Chia dữ liệu:* train/val/test; split chính thức hay tự chia; seed; phân tầng; đơn vị chia; chống leakage; quy tắc chọn tập con.
  - *Tiền xử lý:* làm sạch; chuẩn hóa; augmentation; kích thước tensor đầu vào; một batch mẫu.
- **Phần 2 — Phương pháp** (§3.2)
  - Sơ đồ pipeline (xem A1-01).
  - Mỗi phương pháp: kiến trúc, biểu diễn đầu vào, loss, metric, điểm mạnh và hạn chế, số tham số; phần tự cài đặt và phần dùng thư viện (ghi tên thư viện).
  - Cấu hình huấn luyện (danh sách đầy đủ ở A1-01).
  - Thiết kế thí nghiệm: giả thuyết; yếu tố thay đổi; yếu tố cố định; metric; tiêu chí kết luận.
- **Phần 3 — Kết quả thực nghiệm** (§3.3)
  - Hành vi huấn luyện; kết quả định lượng; kết quả định tính; so sánh và thảo luận; phân tích lỗi; ablation (khuyến khích với A1).
  - *Hạn chế và kết luận:* hạn chế về dữ liệu, phương pháp và thí nghiệm; khả năng tổng quát hóa; hướng phát triển; **câu trả lời ngắn cho câu hỏi nghiên cứu**.

**Theo handbook — yêu cầu thêm**
- Báo cáo phải có mục **AI Usage Disclosure** (§5.1). Danh sách trường thông tin xem ở A1-12.
- Phân biệt rõ phần **bắt buộc** và phần **tùy chọn** (§1.5).
- **Tên file nộp LMS** (§7.1): `<GroupName>_A1_Report.pdf`, trong đó `<GroupName>` phải **trùng khớp chính xác** với Group Name trong sheet `GroupRegistration`.

---

### A1-10 · Slide A1

> **TV B** · P2 · DL nội bộ 19/10 · DL chính thức 21/10
> **Kết quả cần đạt:** slide phủ đủ nội dung mà video phải trình bày

**Theo handbook**
- **Slide là sản phẩm bắt buộc** (§4.1) và phải có link trên trang assignment, ở mục "report/slides" (§2.2).
- Handbook **không quy định định dạng slide**. Slide dùng để hỗ trợ video, mà video phải trình bày (§2.3): bài toán và dữ liệu; phương pháp; thiết lập thí nghiệm; kết quả chính; phân tích lỗi; kết luận; demo ngắn nếu phù hợp.
- **Video không được chỉ đọc lại slide** (§2.3).

---

### A1-11 · Video A1

> **Cả nhóm** · P2 · DL nội bộ 20/10 · DL chính thức 21/10
> **Kết quả cần đạt:** video YouTube Public/Unlisted, mọi thành viên trình bày, đúng format tiêu đề

**Theo handbook** (§2.3)
- **Mỗi bài tập lớn đều bắt buộc có video YouTube.**
- **Format tiêu đề:** `CO3133-Semester-261 – Group [ID] – Assignment 1`
- **Tất cả thành viên đều phải trình bày hoặc bảo vệ một phần nội dung.**
- **Nội dung bắt buộc:** bài toán và dữ liệu; phương pháp; thiết lập thí nghiệm; kết quả chính; phân tích lỗi; kết luận; demo ngắn nếu phù hợp.
- **Không chỉ đọc lại slide.**
- **Chế độ hiển thị:** Public hoặc Unlisted.
- **Nhóm phải tự kiểm tra link trước khi nộp.**

---

### A1-12 · Web A1 & AI disclosure

> **TV C** · P2 · DL nội bộ 20/10 · DL chính thức 21/10
> **Kết quả cần đạt:** `assignment-1.html` đủ mọi mục ở §2.2; AI disclosure ở đủ 4 nơi

**Theo handbook — nội dung tối thiểu của trang assignment** (§2.2)
- [ ] Tên bài tập
- [ ] Thành viên nhóm
- [ ] Giảng viên
- [ ] Phát biểu bài toán
- [ ] Mô tả dataset và EDA
- [ ] Phương pháp
- [ ] Thiết lập thí nghiệm
- [ ] Kết quả
- [ ] So sánh và thảo luận
- [ ] Phân tích lỗi
- [ ] Hạn chế và kết luận
- [ ] Link source code
- [ ] Link checkpoint hoặc hướng dẫn tái tạo checkpoint
- [ ] Link báo cáo/slide
- [ ] Link video YouTube
- [ ] AI Usage Disclosure riêng cho bài tập này

**Theo handbook — landing page** (§1.2, §2.1, §4.1)
- Phải có link đến mọi sản phẩm bắt buộc của từng bài ngay khi sẵn sàng.
- Thông tin nhóm gồm **vai trò hoặc đóng góp chính** của từng thành viên, và link GitHub chỉ khi thực sự có ("không bịa link giả").

**Theo handbook — AI disclosure**
- **Các nơi bắt buộc khai báo** (§5.1, §4.1):
  1. Landing page chung
  2. Trang của từng bài tập
  3. Từng báo cáo
  4. File `AI_USAGE.md` trong repository, có log chi tiết
- **Các trường bắt buộc cho mỗi công cụ AI** (§5.2):
  - tên công cụ và phiên bản/model nếu biết
  - thành viên đã sử dụng
  - thời điểm hoặc giai đoạn phát triển
  - mục đích
  - các phần của bài tập bị ảnh hưởng
  - ví dụ prompt tiêu biểu hoặc link log prompt
  - output AI đã được chỉnh sửa và kiểm chứng như thế nào
  - thành viên chịu trách nhiệm kiểm chứng cuối cùng
  - nguồn dùng để kiểm chứng (tài liệu/bài báo/test)
- **Mẫu khai báo** (§5.3): Tool / Used by / Task / Prompt summary / AI contribution / Student verification / Affected files/sections / Responsible member.
- **Nếu không dùng AI** (§5.4), phải khai báo rõ bằng câu trong handbook: *"The group declares that no generative AI tool was used in this assignment."* (Nhóm cam kết không sử dụng công cụ AI tạo sinh nào trong bài tập này.)
- **Khai báo thiếu hoặc sai có thể bị xem là vi phạm liêm chính học thuật** (§5.5).

---

### A1-13 · Checkpoints & README A1

> **TV A** · P2 · DL nội bộ 19/10 · DL chính thức 21/10
> **Kết quả cần đạt:** link checkpoint + lệnh tái lập kết quả trong README; tag `a1-final`

**Theo handbook**
- **Sản phẩm bắt buộc** (§4.1): checkpoint **hoặc** hướng dẫn tái tạo checkpoint.
- **Repository bắt buộc có** (§4.2):
  - [ ] `README.md` gồm lệnh **cài đặt, chuẩn bị dataset, train và evaluate**
  - [ ] File cấu hình
  - [ ] Thiết lập seed
  - [ ] Phiên bản thư viện phụ thuộc
  - [ ] Thông tin phần cứng
  - [ ] Cách truy cập checkpoint hoặc hướng dẫn tái tạo
  - [ ] Link đến báo cáo và trang assignment
  - [ ] `AI_USAGE.md`
- **Truy vết** (§4.2): kết quả chính → cấu hình mô hình, split, checkpoint, log/mã thí nghiệm, commit/tag.
- **Không được chấp nhận** (§4.2): notebook chạy sẵn mà không có hướng dẫn tái lập.

**Kế hoạch nhóm**
- Git tag `a1-final` là cách nhóm đáp ứng yêu cầu truy vết "commit/tag".

---

### A1-14 · Nộp A1 M2 Final

> **Cả nhóm** · **P0** · DL nội bộ 20/10 · DL chính thức **21/10/2026 23:59**
> **Kết quả cần đạt:** `<GroupName>_A1_Report.pdf` trên LMS + mọi link hoạt động (75% điểm A1)

**Theo handbook**
- **Trọng số** (§7.4): M2 Final = **75% điểm Bài tập lớn 1** (30 điểm đồ án — cột mốc có trọng số lớn nhất).
- **Yêu cầu tối thiểu** (§7.4): **đủ 5 mô hình; so sánh đầy đủ; báo cáo; slide; video YouTube; trang assignment; checkpoint.**
- **Cột mốc Final** (§7.7) phải có **mọi sản phẩm ở §4** và đáp ứng **toàn bộ rubric**.
- **Sản phẩm của mỗi bài** (§4.1):
  1. Trang assignment trên GitHub Pages
  2. Repository source code
  3. Báo cáo (theo §3)
  4. Slide thuyết trình
  5. Video thuyết trình trên YouTube
  6. AI Usage Disclosure (landing page + trang assignment + báo cáo + `AI_USAGE.md`)
  7. Checkpoint hoặc hướng dẫn tái tạo
  8. Báo cáo PDF bản Final nộp trên LMS
- **Nộp LMS** (§7.1):
  - **Đại diện nhóm** nộp file PDF tại <https://lms.hcmut.edu.vn/course/view.php?id=142848>.
  - Tên file `<GroupName>_A1_Report.pdf`, trùng khớp chính xác với Group Name đã đăng ký.
  - Code, video và slide được đặt link trên landing page; chúng **không thay thế** file PDF trên LMS.
- **Nộp trễ** (§7.2): xem mục *Quy tắc chung cho mọi task* ở đầu tài liệu.

**Rubric A1** (§14)

| Hạng mục | Nội dung đánh giá | Trọng số |
|---|---|---|
| Problem & data understanding | Phát biểu bài toán, EDA, split, ý thức về leakage | 15% |
| Data & training pipeline | Dataset/DataLoader, tiền xử lý, vòng lặp train/val/test | 15% |
| Methods & implementation | Đủ 5 mô hình bắt buộc, loss đúng, có giải thích | 30% |
| Experimental design & results | So sánh công bằng, metric ngoài accuracy, đường cong/bảng | 20% |
| Analysis & error analysis | Thảo luận inductive bias, trường hợp sai, hạn chế | 10% |
| Reproducibility, report & presentation | Repo, web, video, AI disclosure, chất lượng trình bày | 10% |

**Kiểm tra trước khi nộp**
- [ ] Đủ 5 mô hình trong phần so sánh (§11.1)
- [ ] Đủ mọi đầu ra ở §12.1
- [ ] File PDF đặt tên đúng, có mục AI disclosure
- [ ] Slide và video có link; video để Public/Unlisted và đã kiểm tra link
- [ ] Trang assignment có đủ mọi mục ở §2.2
- [ ] Có link checkpoint hoặc hướng dẫn tái tạo
- [ ] Lệnh trong README chạy được; kết quả truy được về commit/tag
- [ ] Mọi thành viên đều giải thích được bất kỳ phần nào của bài nộp (§5.5)

---

### A2-04 · Cổng phê duyệt Proposal A2

> **TV A** · **P0** · DL nội bộ 19/10 · DL nhóm tự đặt 21/10\*
> **Kết quả cần đạt:** trạng thái duyệt được ghi vào repo; chưa duyệt thì chưa train chính

**Theo handbook**
- **Các trạng thái** (§18): Approved; Approved with conditions; Rejected.
- **Cổng phê duyệt** (§7.5, §18): chỉ bắt đầu thí nghiệm chính **sau khi được Approved hoặc Approved with conditions**.
- **Chặn tiến độ** (§7.1): thiếu Dataset Proposal sẽ chặn phần triển khai chính cho đến khi proposal được duyệt.
- **Lưu hồ sơ** (§4.1): hồ sơ Dataset Proposal và trạng thái duyệt là sản phẩm bắt buộc của A2.

**Handbook không đề cập**
- Ngày giảng viên duyệt và quy trình nộp lại nếu bị *Rejected*. Nếu gặp trường hợp này, hãy hỏi giảng viên.

**Kế hoạch nhóm**
- \* 21/10 là hạn do nhóm tự đặt, vì baseline A2 bắt đầu train từ 22/10. Nếu trạng thái là "Approved with conditions", phải xử lý các điều kiện trước khi train.

---

# Giai đoạn 5 — A2 M2 Draft (hạn 28/10/2026)

### A2-05 · Baseline A2

> **TV A** · P1 · DL nội bộ 26/10 · DL chính thức 28/10
> **Kết quả cần đạt:** baseline đơn giản đã train + metric trên tập val

**Theo handbook**
- **Bắt buộc** (§20, mục 1): một baseline đơn giản.
- **M2 Draft** (§7.5): baseline đơn giản + mô hình pretrained đang chạy; EDA; metric sơ bộ.
- **Metric** (§20.1): dùng metric bắt buộc của track (danh sách ở A2-01).
- **Mô tả phương pháp trong báo cáo** (§3.2): kiến trúc, biểu diễn đầu vào, loss, metric, điểm mạnh và hạn chế, số tham số, phần tự cài đặt và phần dùng thư viện.
- **Chỉ làm sau khi được duyệt** (§7.5): xem A2-04.

**Ảnh hưởng đến điểm** (§23)
- *Methods & implementation* (25%): baseline, mô hình pretrained/hiện đại, fine-tuning.

---

### A2-06 · Fine-tune pretrained

> **TV B** · P1 · DL nội bộ 26/10 · DL chính thức 28/10
> **Kết quả cần đạt:** quy trình fine-tune được viết rõ (freeze/unfreeze, LR, schedule) + lần chạy đầu tiên

**Theo handbook**
- **Bắt buộc** (§20, mục 2–3): **ít nhất một mô hình pretrained hoặc hiện đại** và **quy trình fine-tuning rõ ràng**.
- **Mục tiêu học tập** (§16): so sánh baseline đơn giản với ít nhất một mô hình pretrained hoặc hiện đại theo **chiến lược fine-tuning có tài liệu hóa**.
- **Phạm vi** (§15.1): ít nhất một mô hình pretrained/hiện đại với quy trình fine-tuning rõ ràng.
- **Cấu hình huấn luyện cần ghi lại** (§3.2): optimizer, learning rate, batch size, số epoch, scheduler, regularization, early stopping, tiêu chí chọn checkpoint, seed, phần cứng, mixed precision, thời gian train, cách chọn siêu tham số.
- **Gợi ý thí nghiệm có kiểm soát** (§20): "freeze vs. full fine-tune" là một ví dụ có sẵn trong handbook (dùng được cho A2-09).

---

### A2-07 · Metrics & web nháp A2

> **TV C** · P1 · DL nội bộ 26/10 · DL chính thức 28/10
> **Kết quả cần đạt:** bảng metric sơ bộ + bản nháp `assignment-2.html` (EDA + metric)

**Theo handbook**
- **Metric** (§20.1): metric bắt buộc của track; task văn bản phải lý giải rõ lựa chọn metric.
- **M2 Draft** (§7.5) cần có **metric sơ bộ** và EDA.
- **Bản nháp** (§7.1, §7.7): có thể đặt link trên landing page; chấp nhận báo cáo chưa hoàn chỉnh và repo đang làm dở.
- **Mục tiêu cho bản Final** (§2.2): đến bản Final, trang A2 phải có đủ danh sách nội dung tối thiểu như ở A1-12.

---

### A2-08 · Nộp A2 M2 Draft

> **Cả nhóm** · **P0** · DL nội bộ 27/10 · DL chính thức **28/10/2026 23:59**
> **Kết quả cần đạt:** baseline + pretrained đang chạy + EDA + metric sơ bộ; có link trên landing page (25% điểm A2)

**Theo handbook**
- **Trọng số** (§7.5): M2 Draft = **25% điểm Bài tập lớn 2**.
- **Yêu cầu tối thiểu** (§7.5): **baseline đơn giản + mô hình pretrained đang chạy; EDA; metric sơ bộ.**
- **Quy định về bản nháp** (§7.1, §7.7): là mốc kiểm tra tiến độ có chấm điểm; có thể đặt link trên landing page; chấp nhận sản phẩm chưa hoàn chỉnh.
- **Nộp trễ** (§7.2): xem mục *Quy tắc chung cho mọi task* ở đầu tài liệu.

**Kiểm tra trước khi nộp**
- [ ] Proposal đã được duyệt (A2-04)
- [ ] Đã có metric của baseline
- [ ] Mô hình pretrained đã bắt đầu train
- [ ] EDA đủ các mục §22 của track
- [ ] Bản nháp có link trên landing page

---

# Giai đoạn 6 — Thí nghiệm A2 và gói nộp Final (29/10 – 11/11/2026)

### A2-09 · Ablation A2

> **TV B** · P1 · DL nội bộ 02/11 · DL chính thức 04/11
> **Kết quả cần đạt:** ≥1 thí nghiệm có kiểm soát, có giả thuyết, yếu tố thay đổi/cố định, kết luận

**Theo handbook**
- **Bắt buộc ít nhất một thí nghiệm có kiểm soát** (§20, mục 4). Ví dụ:
  - freeze vs. full fine-tune
  - có vs. không có augmentation
  - backbone A vs. B
  - loss A vs. B
  - độ phân giải A vs. B
  - một yếu tố kiểm soát khác phù hợp với task
- **Bắt buộc ít nhất một ablation (hoặc nghiên cứu yếu tố có kiểm soát tương đương)** (§21, §3.3).
- **Với mỗi thí nghiệm được báo cáo** (§21): giả thuyết, yếu tố thay đổi, các yếu tố cố định, metric đánh giá, **tiêu chí quyết định**.
- **Báo cáo cho các cấu hình chính** (§21): phần cứng, phiên bản phần mềm, seed, quy tắc chọn checkpoint, thời gian train, chi phí inference.
- **Mục tiêu học tập** (§16): thiết kế và diễn giải ít nhất một ablation hoặc thí nghiệm theo yếu tố có kiểm soát.

**Ảnh hưởng đến điểm** (§23)
- *Experimental design & results* (20%): thí nghiệm có kiểm soát, metric, chi phí tính toán.
- *Analysis & error analysis* (15%): bao gồm ablation.

---

### A2-10 · Định tính & phân tích lỗi A2

> **TV C** (TV A hỗ trợ) · P1 · DL nội bộ 02/11 · DL chính thức 04/11
> **Kết quả cần đạt:** ví dụ đúng/khó/sai + phân nhóm lỗi (tỷ lệ, nguyên nhân)

**Theo handbook**
- **Bắt buộc** (§20, mục 6–7): đánh giá định tính và phân tích lỗi.
- **Bắt buộc ở bản Final** (§7.5): phân tích lỗi nằm trong yêu cầu tối thiểu của M3.
- **Kết quả định tính** (§3.3): dự đoán đúng; trường hợp khó nhưng vẫn đúng; trường hợp sai; so sánh dự đoán với nhãn thật; hình minh họa phù hợp với task.
- **Phân tích lỗi, cho từng nhóm lỗi** (§3.3): mô tả; tần suất/tỷ lệ nếu có thể; ví dụ; nguyên nhân giả định; hướng cải thiện.
- **Mục tiêu học tập** (§16): phân tích lỗi, chi phí tính toán và khả năng tái lập.

**Ảnh hưởng đến điểm** (§23)
- *Analysis & error analysis* (15%): bằng chứng định tính, **phân loại lỗi (error taxonomy)**, ablation.

---

### A2-11 · Compute cost A2

> **TV A** · P1 · DL nội bộ 09/11 · DL chính thức 11/11
> **Kết quả cần đạt:** thời gian train; chi phí inference; phần cứng cho các cấu hình chính

**Theo handbook**
- **Bắt buộc** (§20, mục 8): phân tích chi phí tính toán.
- **Báo cáo cho các cấu hình chính** (§21): phần cứng, phiên bản phần mềm, seed, quy tắc chọn checkpoint, **thời gian train và chi phí inference**.
- **Kết quả định lượng** (§3.3): số tham số; thời gian train/inference; cấu hình.
- **Thảo luận** (§3.3): đánh đổi giữa accuracy – tốc độ – số tham số.

**Ảnh hưởng đến điểm** (§23)
- *Experimental design & results* (20%) có nêu rõ **chi phí tính toán**.

**Kế hoạch nhóm**
- Báo cáo có hạn cùng ngày, nên hãy gửi các số liệu này cho người viết báo cáo trước 09/11.

---

### A2-12 · Báo cáo A2

> **Cả nhóm** · P2 · DL nội bộ 09/11 · DL chính thức 11/11
> **Kết quả cần đạt:** PDF theo §3 + trạng thái duyệt Proposal + AI disclosure

**Theo handbook**
- **Cấu trúc** (§3): gồm 3 phần như báo cáo A1 (xem A1-09).
- **Nội dung riêng của A2:**
  - EDA theo track ở §22 (danh sách ở A2-03).
  - **Bắt buộc ít nhất một ablation** (§3.3, §21).
  - Mỗi thí nghiệm nêu giả thuyết, yếu tố thay đổi, yếu tố cố định, metric, tiêu chí quyết định (§21).
  - Phần cứng, phiên bản phần mềm, seed, quy tắc chọn checkpoint, thời gian train và chi phí inference cho các cấu hình chính (§21).
- **Lưu hồ sơ** (§4.1): có hồ sơ Dataset Proposal và trạng thái duyệt.
- Báo cáo phải có mục **AI Usage Disclosure** (§5.1).
- **Tên file nộp LMS** (§7.1): `<GroupName>_A2_Report.pdf`, trùng khớp chính xác với Group Name đã đăng ký.

---

### A2-13 · Slide A2

> **TV C** · P2 · DL nội bộ 09/11 · DL chính thức 11/11
> **Kết quả cần đạt:** slide phủ đủ nội dung mà video phải trình bày

**Theo handbook**
- Giống A1-10: slide là sản phẩm bắt buộc (§4.1), có link trên trang assignment (§2.2), không quy định định dạng, dùng để hỗ trợ các nội dung video bắt buộc ở §2.3.

---

### A2-14 · Video A2

> **Cả nhóm** · P2 · DL nội bộ 10/11 · DL chính thức 11/11
> **Kết quả cần đạt:** video YouTube Public/Unlisted, mọi thành viên trình bày, đúng format tiêu đề

**Theo handbook**
- Giống A1-11 (§2.3), với tiêu đề `CO3133-Semester-261 – Group [ID] – Assignment 2`.
- Mọi thành viên đều trình bày hoặc bảo vệ một phần nội dung; nhóm tự kiểm tra link trước khi nộp.

---

### A2-15 · Web A2 & AI disclosure

> **TV A** · P2 · DL nội bộ 10/11 · DL chính thức 11/11
> **Kết quả cần đạt:** `assignment-2.html` hoàn chỉnh + link Proposal và trạng thái duyệt; AI disclosure ở đủ 4 nơi

**Theo handbook**
- **Trang assignment**: dùng danh sách 16 mục giống A1-12 (§2.2).
- **Riêng A2** (§4.1): có hồ sơ Dataset Proposal và trạng thái duyệt.
- **AI disclosure**: cùng các nơi khai báo và các trường thông tin như A1-12 (§5.1–§5.5).
- **Landing page** (§4.1): phải có link đến mọi sản phẩm của A2.

---

### A2-16 · Checkpoints & README A2

> **TV B** · P2 · DL nội bộ 09/11 · DL chính thức 11/11
> **Kết quả cần đạt:** link checkpoint + lệnh tái lập; tag `a2-final`

**Theo handbook**
- Dùng cùng checklist repository và quy tắc truy vết như A1-13 (§4.1, §4.2).
- **Bổ sung cho A2** (§21): phải ghi rõ seed, quy tắc chọn checkpoint và phiên bản phần cứng/phần mềm cho các cấu hình chính.

---

### A2-17 · Nộp A2 M3 Final

> **Cả nhóm** · **P0** · DL nội bộ 10/11 · DL chính thức **11/11/2026 23:59**
> **Kết quả cần đạt:** `<GroupName>_A2_Report.pdf` trên LMS + mọi link hoạt động (60% điểm A2)

**Theo handbook**
- **Trọng số** (§7.5): M3 Final = **60% điểm Bài tập lớn 2** (18 điểm đồ án).
- **Yêu cầu tối thiểu** (§7.5): **thí nghiệm có kiểm soát; phân tích lỗi; báo cáo; slide; video; trang assignment; checkpoint.**
- **Cột mốc Final** (§7.7): đủ mọi sản phẩm ở §4 + toàn bộ rubric. A2 cần thêm hồ sơ Dataset Proposal và trạng thái duyệt (§4.1).
- **Nộp LMS** (§7.1): đại diện nhóm nộp `<GroupName>_A2_Report.pdf`; các file được đặt link không thay thế file này.
- **Nhắc lại phạm vi** (§15.1): phải hoàn thành đủ phần bắt buộc; phần mở rộng không thay thế được phần bắt buộc còn thiếu.
- **Nộp trễ** (§7.2): xem mục *Quy tắc chung cho mọi task* ở đầu tài liệu.

**Rubric A2** (§23)

| Hạng mục | Nội dung đánh giá | Trọng số |
|---|---|---|
| Problem & data understanding | Chất lượng proposal, EDA, kiểm soát split/leakage | 15% |
| Data & training pipeline | Nạp dữ liệu quy mô lớn, tiền xử lý, vòng lặp train | 15% |
| Methods & implementation | Baseline, mô hình pretrained/hiện đại, fine-tuning | 25% |
| Experimental design & results | Thí nghiệm có kiểm soát, metric, chi phí tính toán | 20% |
| Analysis & error analysis | Bằng chứng định tính, phân loại lỗi, ablation | 15% |
| Reproducibility, report & presentation | Repo, web, video, AI disclosure, trình bày | 10% |

**Kiểm tra trước khi nộp**
- [ ] Đã so sánh baseline với mô hình pretrained theo quy trình fine-tuning được mô tả rõ (§20)
- [ ] ≥1 ablation có giả thuyết / các yếu tố / tiêu chí quyết định (§21)
- [ ] Có đánh giá định lượng + định tính, phân tích lỗi, phân tích chi phí tính toán (§20)
- [ ] File PDF đặt tên đúng, có AI disclosure và trạng thái duyệt Proposal
- [ ] Slide và video có link; video để Public/Unlisted và đã kiểm tra link
- [ ] Trang assignment hoàn chỉnh, có hồ sơ Proposal và trạng thái duyệt
- [ ] Có checkpoint hoặc hướng dẫn tái tạo; lệnh trong README chạy được; kết quả truy được về commit/tag
- [ ] Mọi thành viên đều giải thích được bất kỳ phần nào của bài nộp (§5.5)
