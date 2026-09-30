---
name: ui-asset-extractor-task
description: Process pending UIAssetExtractor tasks (analyze/QA) in the tool's data\cowork_tasks folder when the user says 'Làm task UIAssetExtractor'.
---

# UIAssetExtractor – xử lý task Cowork

Tool UIAssetExtractor (web local của người dùng) ở chế độ "Claude app (Cowork)" không gọi Claude API mà tạo task và chờ `result.json`.

## Các bước

1. Tìm thư mục tool: kiểm tra các thư mục đã kết nối (get_device_info). Thư mục tool là thư mục chứa `start.bat` và `app\`, tên thường là `UIAssetExtractor` (mặc định `E:\Tool\UIAssetExtractor`). Nếu chưa kết nối, hỏi người dùng tool nằm ở đâu rồi xin quyền truy cập bằng device_request_folder_access.
2. Liệt kê `<thư mục tool>\data\cowork_tasks`. Task đang chờ = thư mục KHÔNG bắt đầu bằng `done_` hoặc `old_` và chưa có `result.json`. Nếu không có task nào: báo người dùng bấm bước Phân tích AI / QA trong tool trước.
3. Với mỗi task đang chờ (thường chỉ 1): stage `TASK.md`, `meta.json` và toàn bộ file .png vào cloud workspace (tối đa 50 file mỗi lần stage), đọc `TASK.md` và làm ĐÚNG theo nó.
   - Task `analyze_…`: xem `screen_marked.png` rồi từng `group_<i>.png` (có lưới 0–1000). Nên ghép vài crop thành 1 ảnh sheet để xem nhanh, nhưng toạ độ bbox phải đọc theo lưới của từng crop riêng. Phần tử UI rõ ràng nhưng không nằm trong vùng nào thì đưa vào "missing" (bbox chuẩn hoá theo toàn ảnh).
   - Task `qa_…`: xem từng `qa_<id>.png` (trái = gốc, phải = asset trên nền caro). Chấm điểm trung thực.
4. Viết `result.json` đúng schema trong TASK.md (chỉ JSON, UTF-8), có đủ mọi id. Ghi file vào `/mnt/user-data/outputs/…` rồi dùng device_commit_files ghi vào đúng thư mục task trên máy người dùng. Ghi một lần, đầy đủ.
5. Nếu sau vài giây trong thư mục xuất hiện `result_error.txt`, đọc lỗi, sửa JSON và ghi lại `result.json`. Thư mục task được đổi tên thành `done_…` nghĩa là tool đã nhận kết quả.
6. Báo người dùng 1–2 câu: đã xử lý task nào, bao nhiêu asset / điểm QA thấp nào; tool sẽ tự chạy tiếp.

## Lưu ý chất lượng (analyze)

- Badge/sticker đè lên icon ("!", "NEW", "200%") là asset riêng, z cao hơn.
- Icon và name plate bên dưới là 2 asset riêng; plate: remove_text=true, nine_slice=true.
- Nút có chữ: remove_text=true. Chữ nằm trực tiếp trên nền: type "text", export=false.
- Quảng cáo, thanh hệ thống, phần tử bị cắt mép ảnh, hoạ tiết nền: keep=false.
- bbox ôm sát pixel; description mô tả hình dạng/màu/viền đủ để vẽ lại, không nhắc nội dung chữ.
