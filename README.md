# UI Asset Extractor

Tool chạy local trên web: từ 1 screenshot UI game → tách thành từng asset PNG trong suốt + sprite sheet (JSON kiểu TexturePacker).

## Chạy

1. Double-click **`start.bat`**.
   - Lần đầu tool tự cài mọi thứ vào thư mục `.runtime` (Python 3.11, thư viện, ONNX Runtime bản GPU nếu có card NVIDIA). Mất khoảng 5–15 phút tuỳ mạng.
   - Sau đó trình duyệt tự mở `http://127.0.0.1:8765`.
2. Vào **Settings**:
   - **Cách dùng Claude**: mặc định là **Claude app (Cowork)**, không cần API key (xem bên dưới). Có API key thì chọn "Claude API key" rồi dán key.
   - **OpenAI**: bấm **Lấy key**, dán key, bấm **Kiểm tra & lưu**. Chỉ cần cho bước Vẽ lại GPT.
3. Kéo thả screenshot vào trang Projects, bấm **⚡ Tự động**.

Tắt tool: đóng cửa sổ console.

## Chế độ Claude app (Cowork)

Ở bước **Phân tích AI** và **QA**, tool không gọi API mà tạo một task trong
`data\cowork_tasks\<tên task>\` (gồm ảnh và `TASK.md`), rồi chờ.

1. Mở Claude trong app Cowork, nhớ kết nối thư mục `E:\Tool`.
2. Nhắn: **"Làm task UIAssetExtractor"**.
3. Claude đọc ảnh và ghi `result.json` vào thư mục task. Tool tự nhận kết quả và chạy tiếp, không cần bấm gì.

Skill cho Claude nằm trong thư mục `skills/` (xem `skills/README.md` để cài vào tài khoản Claude).

Muốn huỷ, bấm **Huỷ** trên thanh tiến độ. Task đã xong được đổi tên thành `done_…`, task bị huỷ thành `old_…`.

## Các bước trong project

| Bước | Làm gì | Tốn API? |
|---|---|---|
| 1 Detect | OpenCV tìm vùng UI + OCR đọc chữ | Không |
| 2 Phân tích AI | Claude đặt tên, tách icon / badge / nút / nhãn, đánh dấu chữ cần xoá, 9-slice | Claude (Cowork hoặc API) |
| 3 Trích xuất | Tách nền (GrabCut hoặc AI rembg), bỏ phần bị đè, xoá chữ, cắt sát, tìm trùng lặp | Không |
| 4 QA | Claude so sánh bản gốc với asset, chấm 0–10 | Claude (Cowork hoặc API) |
| 5 Vẽ lại GPT | Model ảnh OpenAI vẽ lại asset tick chọn (nền trong suốt, không chữ, bù phần bị che), tự QA và thử lại | OpenAI |
| 6 Export | `atlas_N.png` + `atlas_N.json` + `sprites/*.png` + `manifest.json` + file .zip | Không |

Không có key nào vẫn dùng được: bước 1 → 2 (Cowork) → 3 → 6.

Thao tác trên ảnh: kéo box để di chuyển, kéo góc dưới phải để đổi kích thước, `B` vẽ box mới, `Del` xoá, `Ctrl + lăn chuột` zoom.

## GPU

- Settings → Chế độ GPU: **Tắt / Tiết kiệm (≤2 GB VRAM, mặc định) / Đầy đủ**.
- Mỗi lúc chỉ nạp 1 model và giải phóng sau mỗi tác vụ. OCR luôn chạy CPU.
- GPU chỉ dùng cho tách nền AI (rembg). GrabCut và các bước khác chạy CPU rất nhẹ.
- Driver NVIDIA ≥ 580 → CUDA 13; 528–579 → CUDA 12; thấp hơn → CPU. Lỗi GPU: chạy `setup.bat cpu`.

## Mang sang máy khác

Clone từ git (hoặc copy cả thư mục **trừ** `.runtime` và `data`), rồi chạy `start.bat` trên máy mới. `.gitignore` đã loại sẵn `.runtime/`, `data/`, `models/`. Chỉ cần Windows 10/11 và Internet cho lần cài đầu. API key lưu trong Windows Credential Manager của từng máy, nên cần nhập lại.

## Thư mục

```
start.bat / setup.bat   chạy tool / cài (lại) môi trường
skills/                 skill cho Claude (Cowork)
app/                    backend FastAPI + pipeline xử lý ảnh
web/                    giao diện (HTML/JS thuần, không cần build)
data/projects/<id>/     ảnh gốc, crops, assets, regen, export
models/                 model AI tải về (rembg)
.runtime/               Python + thư viện (tự tạo, xoá được)
```

Sửa lỗi cài đặt: xoá `.runtime` rồi chạy lại `setup.bat`.
