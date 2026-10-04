# Failure Analysis — Lab 18: Production RAG

Học viên: Trương Việt Anh 
MSSV: 2A202602444 
Khóa K4, Track 3A


## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| faithfulness | 0.8833 | 0.9429 | +0.0595 |
| answer_relevancy | 0.7416 | 0.9040 | +0.1623 |
| context_precision | 0.7000 | 0.9000 | +0.2000 |
| context_recall | 0.8500 | 0.9333 | +0.0833 |

## Bottom-5 Failures

### #1

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Theo chính sách nghỉ phép năm phiên bản 2024 (có hiệu lực từ 01/01/2024):

*   **Số ngày phép năm:** Nhân viên có 9 năm thâm niên được hưởng **18 ngày phép năm** (15 ngày tiêu chuẩn + 3 ngày thâm niên, tính theo công thức: 1 ngày phép cho mỗi 3 năm làm việc liên tục).
*   **Lương:** Tài liệu không cung cấp thông tin về mức lương hoặc khoảng lương của nhân viên.
- **Worst metric:** context_precision = 0.0000
- **Câu trả lời đúng không?** Một phần. Câu trả lời chính xác về số ngày phép (18 ngày) dựa trên chính sách 2024, nhưng thiếu thông tin về mức lương trong khi ground_truth có đề cập đến khoảng lương 20-35 triệu VNĐ.
- **Context chứa đáp án không?** Thiếu. Các ngữ cảnh được cung cấp (nghi_phep_nam_v2024.md, nghi_phep_nam_v2023.md, nghi_phep_khong_luong.md) hoàn toàn không chứa thông tin về mức lương của nhân viên Senior.
- **Cần viết lại câu hỏi không?** Câu hỏi rõ; có thể phân rã thành hai truy vấn con về phép năm theo thâm niên và lương Senior, rồi hợp nhất bằng chứng.
- **Module cần sửa:** M2 retrieval và M3 chọn context: cần phủ đủ hai khía cạnh của câu hỏi multi-hop, không chỉ các tài liệu nghỉ phép.
- **Error Tree:** Output đúng? Một phần → Context đủ? Thiếu bảng lương → Query rõ? Có, hai khía cạnh → Fix module: M2/M3.
- **Root cause:** Bảng lương có trong corpus: `data/bang_luong_2024.md`, dòng 8 ghi Senior (P3-P4) 20.000.000–35.000.000. Nhưng bảng này không nằm trong ba context được chọn. Chưa đủ bằng chứng xác định nó bị loại ở retrieval hay reranking; cần xem các candidate trung gian. Không được suy từ thiếu context thành thiếu dữ liệu nguồn.
- **Suggested fix:** Ghi log candidate trước/sau rerank; thử phân rã multi-hop và chọn context đa dạng theo khía cạnh. Giữ bảng lương hiện có, không tạo tài liệu thay thế. Chạy lại toàn bộ 20 câu sau thay đổi để kiểm tra hồi quy.

**Context thực tế đã gửi cho mô hình:**

> Nguồn: nghi_phep_nam_v2024.md
> # Chính sách nghỉ phép năm (Phiên bản 2024)
> > Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự
>
> ## Số ngày phép năm
> Mỗi nhân viên chính thức được hưởng **15 ngày phép năm** có lương, tăng từ 12 ngày so với chính sách năm 2023. Chính sách này thay thế hoàn toàn phiên bản 1.0 ban hành ngày 01/01/2023.
>
> ## Thâm niên công tác
> Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 3 năm làm việc liên tục. Ví dụ: nhân viên 9 năm thâm niên được 18 ngày phép (15 + 3).
>
> ## Quy định sử dụng
> Phép năm phải được đăng ký trước ít nhất 2 ngày làm việc qua hệ thống HR Portal. Nghỉ phép từ 5 ngày trở lên cần được trưởng phòng phê duyệt. Phép năm không sử dụng hết có thể chuyển tối đa 5 ngày sang năm sau, hết hạn vào 31/03 năm kế tiếp. Nhân viên cũng có thể yêu cầu thanh toán bằng tiền cho số ngày phép còn lại.

> Nguồn: nghi_phep_nam_v2023.md
> # Chính sách nghỉ phép năm (Phiên bản 2023)
> > Phiên bản: 1.0 | Ngày hiệu lực: 01/01/2023 | Phòng ban: Nhân sự
>
> ## Số ngày phép năm
> Mỗi nhân viên chính thức được hưởng **12 ngày phép năm** có lương. Số ngày phép được tính theo năm dương lịch từ 01/01 đến 31/12. Nhân viên vào công ty giữa năm sẽ được tính pro-rata theo số tháng làm việc thực tế.
>
> ## Thâm niên công tác
> Nhân viên có thâm niên từ **5 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 5 năm làm việc liên tục. Ví dụ: nhân viên 10 năm thâm niên được 14 ngày phép.
>
> ## Quy định sử dụng
> Phép năm phải được đăng ký trước ít nhất 3 ngày làm việc qua hệ thống HR Portal. Nghỉ phép từ 3 ngày trở lên cần được trưởng phòng phê duyệt. Phép năm không sử dụng hết có thể chuyển tối đa 5 ngày sang năm sau, hết hạn vào 31/03 năm kế tiếp.

> Nguồn: nghi_phep_khong_luong.md
> # Chính sách nghỉ phép không lương
> > Phiên bản: 1.2 | Ngày hiệu lực: 01/07/2023 | Phòng ban: Nhân sự
>
> ## Điều kiện áp dụng
> Nhân viên chính thức đã hết phép năm có thể xin nghỉ phép không lương. Thời gian nghỉ phép không lương tối đa là **30 ngày** trong một năm dương lịch. Nghỉ phép không lương không được tính thâm niên.
>
> ## Quy trình phê duyệt
> Đơn xin nghỉ phép không lương phải được nộp trước ít nhất 7 ngày làm việc. Nghỉ từ 1-5 ngày: trưởng phòng phê duyệt. Nghỉ từ 6-15 ngày: cần thêm phê duyệt của Giám đốc Nhân sự. Nghỉ từ 16-30 ngày: cần phê duyệt của **Giám đốc điều hành (CEO)**.
>
> ## Ảnh hưởng đến phúc lợi
> Trong thời gian nghỉ không lương, nhân viên không được hưởng lương và các khoản phụ cấp. Bảo hiểm y tế vẫn được duy trì nếu nghỉ dưới 14 ngày. Nghỉ trên 14 ngày, nhân viên cần tự đóng phần bảo hiểm của mình.

### #2

- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** Có, tất cả nhân viên bắt buộc kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ. Các phương thức MFA được chấp nhận bao gồm: ứng dụng Authenticator (ưu tiên), SMS OTP, hoặc YubiKey.
- **Worst metric:** context_recall = 0.5000
- **Câu trả lời đúng không?** Đúng. Câu trả lời đã trích xuất chính xác thông tin từ tài liệu 'mat_khau_v2.md' về việc bắt buộc kích hoạt MFA và các phương thức được chấp nhận.
- **Context chứa đáp án không?** Đủ. Đoạn văn bản trong 'mat_khau_v2.md' tại mục 'Xác thực đa yếu tố (MFA)' chứa đầy đủ thông tin cần thiết để trả lời câu hỏi.
- **Cần viết lại câu hỏi không?** Không cần viết lại. Câu hỏi đã rõ ràng, trực diện và hệ thống đã truy xuất được tài liệu chứa câu trả lời chính xác.
- **Module cần sửa:** Không cần sửa. Hệ thống đã hoạt động tốt trong việc truy xuất và tổng hợp thông tin từ ngữ cảnh phù hợp.
- **Error Tree:** Output đúng → Context đủ → Query rõ → Không cần sửa module.
- **Root cause:** Câu trả lời đúng yêu cầu hiện hành. Ground truth còn nêu chính sách cũ v1.0; context thiếu đối chiếu lịch sử này có thể góp phần làm recall thấp. Đây là giả thuyết cần kiểm tra bằng trace chấm M4, không phải bằng chứng câu trả lời sai.
- **Suggested fix:** Kiểm tra mức phù hợp của ground truth với phạm vi câu hỏi ở M4; không thay điểm hoặc sửa ground truth chỉ để tăng điểm. Khi người dùng hỏi so sánh phiên bản, cần truy xuất cả hai bản.

**Context thực tế đã gửi cho mô hình:**

> Nguồn: mat_khau_v2.md
> # Chính sách mật khẩu (Phiên bản hiện hành)
> > Phiên bản: 2.0 | Ngày hiệu lực: 01/07/2024 | Phòng ban: CNTT
>
> ## Yêu cầu mật khẩu
> Mật khẩu phải có tối thiểu **12 ký tự**, bao gồm ít nhất 1 chữ hoa, 1 chữ thường, 1 số và 1 ký tự đặc biệt (!@#$%^&*). Khuyến khích sử dụng passphrase dài hơn 16 ký tự.
>
> ## Xác thực đa yếu tố (MFA)
> Tất cả nhân viên **bắt buộc** kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ. Phương thức MFA được chấp nhận: ứng dụng Authenticator (ưu tiên), SMS OTP, hoặc YubiKey.
>
> ## Chu kỳ thay đổi
> Mật khẩu phải được thay đổi **mỗi 120 ngày**. Hệ thống tự động nhắc nhở trước 14 ngày. Mật khẩu mới không được trùng với 5 mật khẩu gần nhất. Tài khoản bị khóa sau 5 lần nhập sai liên tiếp.
>
> ## Chính sách thay thế
> Văn bản này thay thế Chính sách mật khẩu v1.0 ban hành ngày 01/01/2022.

> Nguồn: mua_sam.md
> # Quy trình mua sắm
> > Phiên bản: 2.2 | Ngày hiệu lực: 01/04/2024 | Phòng ban: Hành chính & Tài chính
>
> ## Thẩm quyền phê duyệt
> | Giá trị đơn hàng | Người phê duyệt |
> |-------------------|-----------------|
> | Dưới **5.000.000 VNĐ** | Trưởng phòng (Manager) |
> | Từ **5.000.000 - 50.000.000 VNĐ** | Giám đốc phòng ban (Director) |
> | Trên **50.000.000 VNĐ** | Tổng Giám đốc (CEO) |
>
> ## Quy trình đề xuất
> 1. Tạo phiếu đề xuất mua sắm trên hệ thống Procurement Portal
> 2. Đính kèm ít nhất 3 báo giá cho đơn hàng trên 10.000.000 VNĐ
> 3. Chờ phê duyệt theo thẩm quyền tương ứng
> 4. Phòng Mua sắm đặt hàng và theo dõi giao nhận
>
> ## Lưu ý đặc biệt
> Mua sắm thiết bị CNTT (laptop, server, phần mềm) cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất. Đơn hàng khẩn cấp có thể bỏ qua yêu cầu 3 báo giá nhưng phải có giải trình bằng văn bản.

> Nguồn: vpn_truy_cap.md
> # Chính sách VPN và truy cập từ xa
> > Phiên bản: 1.3 | Ngày hiệu lực: 01/04/2024 | Phòng ban: CNTT
>
> ## Giao thức kết nối
> Công ty sử dụng **WireGuard VPN** với mã hóa **AES-256**. Tất cả kết nối từ xa bắt buộc qua VPN. Nhân viên tải cấu hình VPN từ IT Portal hoặc liên hệ helpdesk@cty.vn.
>
> ## Quản lý tài khoản VPN
> Tài khoản VPN tự động bị **khóa sau 60 ngày không hoạt động**. Để kích hoạt lại, nhân viên cần gửi yêu cầu qua ticket IT kèm xác nhận của quản lý trực tiếp. Thời gian xử lý: 1-2 ngày làm việc.
>
> ## Quy định sử dụng
> - Không chia sẻ thông tin đăng nhập VPN
> - Ngắt kết nối VPN khi không sử dụng
> - Không sử dụng VPN cá nhân (NordVPN, ExpressVPN) trong mạng công ty
> - Phiên VPN tự động ngắt sau 12 giờ, cần đăng nhập lại

### #3

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Theo chính sách nghỉ phép năm mới nhất (Phiên bản 2024, hiệu lực từ 01/01/2024), nhân viên chính thức được hưởng **15 ngày phép năm** có lương.

Ngoài ra, nhân viên có thể được cộng thêm ngày phép dựa trên thâm niên công tác:
- Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 3 năm làm việc liên tục (Ví dụ: nhân viên 9 năm thâm niên được 18 ngày phép).
- **Worst metric:** context_precision = 0.5000
- **Câu trả lời đúng không?** Đúng. Câu trả lời phản ánh chính xác thông tin từ tài liệu v2024, bao gồm cả số ngày phép cơ bản và quy định thâm niên.
- **Context chứa đáp án không?** Đủ. Thông tin nằm trong file 'nghi_phep_nam_v2024.md' với nội dung: 'Mỗi nhân viên chính thức được hưởng 15 ngày phép năm có lương'.
- **Cần viết lại câu hỏi không?** Không cần viết lại. Câu hỏi đã đủ rõ ràng và trực diện, hệ thống đã truy xuất được đúng tài liệu cần thiết.
- **Module cần sửa:** Không cần sửa đáp án; nên kiểm tra M2/M3 vì context_precision 0.5 cho thấy còn ngữ cảnh dư hoặc ít liên quan.
- **Error Tree:** Output đúng (Đúng) → Context đủ (Đủ) → Query rõ (Rõ) → Fix module (Không cần sửa).
- **Root cause:** Generation chọn đúng bản 2024 nhưng retrieval vẫn đưa bản cũ và tài liệu nghỉ không lương vào context. Chưa thể kết luận pipeline có bộ lọc hiệu lực tự động chỉ từ một câu trả lời đúng.
- **Suggested fix:** Bổ sung metadata hiệu lực và thử lọc phiên bản cho câu hỏi hiện hành; giữ bản cũ khi cần so sánh lịch sử. Đánh giá lại precision trên toàn bộ bộ câu hỏi.

**Context thực tế đã gửi cho mô hình:**

> Nguồn: nghi_phep_nam_v2023.md
> # Chính sách nghỉ phép năm (Phiên bản 2023)
> > Phiên bản: 1.0 | Ngày hiệu lực: 01/01/2023 | Phòng ban: Nhân sự
>
> ## Số ngày phép năm
> Mỗi nhân viên chính thức được hưởng **12 ngày phép năm** có lương. Số ngày phép được tính theo năm dương lịch từ 01/01 đến 31/12. Nhân viên vào công ty giữa năm sẽ được tính pro-rata theo số tháng làm việc thực tế.
>
> ## Thâm niên công tác
> Nhân viên có thâm niên từ **5 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 5 năm làm việc liên tục. Ví dụ: nhân viên 10 năm thâm niên được 14 ngày phép.
>
> ## Quy định sử dụng
> Phép năm phải được đăng ký trước ít nhất 3 ngày làm việc qua hệ thống HR Portal. Nghỉ phép từ 3 ngày trở lên cần được trưởng phòng phê duyệt. Phép năm không sử dụng hết có thể chuyển tối đa 5 ngày sang năm sau, hết hạn vào 31/03 năm kế tiếp.

> Nguồn: nghi_phep_nam_v2024.md
> # Chính sách nghỉ phép năm (Phiên bản 2024)
> > Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự
>
> ## Số ngày phép năm
> Mỗi nhân viên chính thức được hưởng **15 ngày phép năm** có lương, tăng từ 12 ngày so với chính sách năm 2023. Chính sách này thay thế hoàn toàn phiên bản 1.0 ban hành ngày 01/01/2023.
>
> ## Thâm niên công tác
> Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 3 năm làm việc liên tục. Ví dụ: nhân viên 9 năm thâm niên được 18 ngày phép (15 + 3).
>
> ## Quy định sử dụng
> Phép năm phải được đăng ký trước ít nhất 2 ngày làm việc qua hệ thống HR Portal. Nghỉ phép từ 5 ngày trở lên cần được trưởng phòng phê duyệt. Phép năm không sử dụng hết có thể chuyển tối đa 5 ngày sang năm sau, hết hạn vào 31/03 năm kế tiếp. Nhân viên cũng có thể yêu cầu thanh toán bằng tiền cho số ngày phép còn lại.

> Nguồn: nghi_phep_dac_biet.md
> # Chính sách nghỉ phép đặc biệt
> > Phiên bản: 1.1 | Ngày hiệu lực: 01/03/2024 | Phòng ban: Nhân sự
>
> ## Các trường hợp được nghỉ phép đặc biệt
> Nhân viên được nghỉ có lương trong các trường hợp sau đây mà không trừ vào phép năm:
> - **Kết hôn**: 3 ngày làm việc
> - **Tang lễ** (cha mẹ, vợ/chồng, con): 3 ngày làm việc
> - **Sinh con** (nam giới): 5 ngày làm việc liên tục
> - **Con kết hôn**: 1 ngày làm việc
>
> ## Hồ sơ cần thiết
> Nhân viên cần cung cấp giấy tờ chứng minh trong vòng 5 ngày làm việc sau khi quay lại: giấy đăng ký kết hôn, giấy khai tử, giấy khai sinh. Trường hợp không cung cấp đầy đủ, số ngày nghỉ sẽ bị trừ vào phép năm.
>
> ## Lưu ý
> Phép đặc biệt không cộng dồn và không chuyển sang năm sau. Ngày nghỉ được tính liên tục bao gồm cả ngày nghỉ cuối tuần nếu rơi vào giữa kỳ nghỉ.

### #4

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ, tương đương **25.000.000 VNĐ**.

**Lý do:** Theo chính sách hoàn chi đào tạo, nhân viên phải cam kết làm việc tại công ty ít nhất 1 năm sau khi hoàn thành khóa học. Nếu nghỉ việc trước thời hạn cam kết (trong trường hợp này là 8 tháng), nhân viên phải hoàn trả 100% chi phí đã được tài trợ.
- **Worst metric:** faithfulness = 0.4000
- **Câu trả lời đúng không?** Đúng. Câu trả lời xác định chính xác số tiền hoàn trả là 25.000.000 VNĐ (100%) dựa trên việc nhân viên nghỉ việc sau 8 tháng, vi phạm cam kết làm việc 1 năm theo chính sách.
- **Context chứa đáp án không?** Đủ. Đoạn văn trong 'hoan_chi_dao_tao.md' nêu rõ: 'Nhân viên được tài trợ đào tạo phải cam kết làm việc tại công ty ít nhất 1 năm sau khi hoàn thành khóa học. Nếu nghỉ việc trước thời hạn cam kết, nhân viên phải hoàn trả 100% chi phí'.
- **Cần viết lại câu hỏi không?** Không cần viết lại. Câu hỏi đã đủ rõ ràng về dữ kiện (số tiền, thời gian hoàn thành, thời gian nghỉ việc) để truy xuất thông tin từ ngữ cảnh.
- **Module cần sửa:** Không cần sửa. Hệ thống đã truy xuất đúng ngữ cảnh và đưa ra câu trả lời chính xác dựa trên dữ liệu cung cấp.
- **Error Tree:** Output đúng → Context đủ → Query rõ → Không cần sửa module.
- **Root cause:** Đáp án khớp ground truth và quy tắc hoàn trả. Số tiền 25 triệu, thời gian 8 tháng là dữ kiện trong câu hỏi, không phải nguyên văn tài liệu. Faithfulness 0.4 cần xem trace tách claim/chấm entailment; chưa đủ bằng chứng khẳng định hallucination hay kết luận evaluator sai.
- **Suggested fix:** Kiểm tra M4 theo từng claim, phân biệt dữ kiện người dùng với bằng chứng chính sách; generation nên trình bày rõ phép suy luận 8 tháng < 12 tháng, hoàn trả 100% × 25 triệu. Giữ nguyên điểm đo gốc.

**Context thực tế đã gửi cho mô hình:**

> Nguồn: hoan_chi_dao_tao.md
> # Chính sách hoàn chi đào tạo
> > Phiên bản: 1.1 | Ngày hiệu lực: 01/07/2023 | Phòng ban: Nhân sự & Tài chính
>
> ## Điều kiện được tài trợ
> Công ty chi trả chi phí đào tạo bên ngoài (chứng chỉ, khóa học dài hạn, hội thảo) cho nhân viên có **thâm niên từ 1 năm trở lên**. Chi phí được tài trợ tối đa **30.000.000 VNĐ/khóa** và cần phê duyệt của Giám đốc phòng ban.
>
> ## Cam kết hoàn chi
> Nhân viên được tài trợ đào tạo phải **cam kết làm việc tại công ty ít nhất 1 năm** sau khi hoàn thành khóa học. Nếu nghỉ việc trước thời hạn cam kết, nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ.
>
> ## Quy trình
> 1. Nộp đơn đề xuất đào tạo kèm chương trình chi tiết
> 2. Trưởng phòng đánh giá mức độ liên quan đến công việc
> 3. Phòng Nhân sự xem xét ngân sách
> 4. Ký cam kết hoàn chi trước khi đăng ký khóa học

> Nguồn: dao_tao_noi_bo.md
> # Chính sách đào tạo nội bộ
> > Phiên bản: 1.3 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự & Đào tạo
>
> ## Yêu cầu đào tạo
> Mỗi phòng ban phải tổ chức tối thiểu **2 buổi đào tạo/quý** về chuyên môn hoặc kỹ năng mềm. Nhân viên mới (trong 6 tháng đầu) **bắt buộc** tham gia tất cả các buổi đào tạo của phòng ban.
>
> ## Ngân sách đào tạo
> Công ty dành ngân sách đào tạo **5.000.000 VNĐ/người/năm** cho các khóa học bên ngoài. Nhân viên có thể đăng ký khóa học qua hệ thống Learning Portal và cần phê duyệt của trưởng phòng.
>
> ## Đánh giá hiệu quả
> Sau mỗi khóa đào tạo, nhân viên cần hoàn thành bài đánh giá và khảo sát phản hồi. Phòng Đào tạo tổng hợp kết quả hàng quý để cải thiện chương trình đào tạo.

> Nguồn: nghi_phep_nam_v2024.md
> # Chính sách nghỉ phép năm (Phiên bản 2024)
> > Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự
>
> ## Số ngày phép năm
> Mỗi nhân viên chính thức được hưởng **15 ngày phép năm** có lương, tăng từ 12 ngày so với chính sách năm 2023. Chính sách này thay thế hoàn toàn phiên bản 1.0 ban hành ngày 01/01/2023.
>
> ## Thâm niên công tác
> Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 3 năm làm việc liên tục. Ví dụ: nhân viên 9 năm thâm niên được 18 ngày phép (15 + 3).
>
> ## Quy định sử dụng
> Phép năm phải được đăng ký trước ít nhất 2 ngày làm việc qua hệ thống HR Portal. Nghỉ phép từ 5 ngày trở lên cần được trưởng phòng phê duyệt. Phép năm không sử dụng hết có thể chuyển tối đa 5 ngày sang năm sau, hết hạn vào 31/03 năm kế tiếp. Nhân viên cũng có thể yêu cầu thanh toán bằng tiền cho số ngày phép còn lại.

### #5

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Dựa trên tài liệu cung cấp:

*   **Người phê duyệt:** Giám đốc phòng ban (Director) vì đơn hàng có giá trị 30.000.000 VNĐ (nằm trong khoảng từ 5.000.000 - 50.000.000 VNĐ).
*   **Yêu cầu từ phòng CNTT:** Cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất.
- **Worst metric:** faithfulness = 0.6000
- **Câu trả lời đúng không?** Một phần. Câu trả lời xác định đúng người phê duyệt và yêu cầu xác nhận từ phòng CNTT, nhưng thiếu thông tin quan trọng về việc cần đính kèm ít nhất 3 báo giá cho đơn hàng trên 10 triệu VNĐ theo quy định tại tài liệu mua_sam.md.
- **Context chứa đáp án không?** Đủ. Tài liệu 'mua_sam.md' cung cấp đầy đủ thông tin về thẩm quyền phê duyệt (Giám đốc), yêu cầu xác nhận từ phòng CNTT và quy định đính kèm 3 báo giá cho đơn hàng trên 10 triệu.
- **Cần viết lại câu hỏi không?** Không cần viết lại. Câu hỏi đã đủ rõ ràng để truy xuất thông tin từ ngữ cảnh.
- **Module cần sửa:** Generation trong `src/pipeline.py`, không phải M5 enrichment. Context đã có quy định ba báo giá nhưng câu trả lời chưa nêu điều kiện bổ sung này.
- **Error Tree:** Output đúng (Thiếu chi tiết báo giá) → Context đủ (Có thông tin 3 báo giá) → Query rõ (Không cần sửa) → Fix module (Generation).
- **Root cause:** Câu trả lời thiếu một điều kiện quy trình so với ground truth; không có bằng chứng bịa thông tin. Điểm faithfulness thấp vẫn cần kiểm tra riêng các claim, vì bỏ sót chi tiết không tự động chứng minh thiếu căn cứ.
- **Suggested fix:** Cải thiện prompt cho bước Generation để yêu cầu mô hình liệt kê đầy đủ các điều kiện/quy trình liên quan đến giá trị đơn hàng được đề cập trong ngữ cảnh, thay vì chỉ tập trung vào người phê duyệt.

**Context thực tế đã gửi cho mô hình:**

> Nguồn: mua_sam.md
> # Quy trình mua sắm
> > Phiên bản: 2.2 | Ngày hiệu lực: 01/04/2024 | Phòng ban: Hành chính & Tài chính
>
> ## Thẩm quyền phê duyệt
> | Giá trị đơn hàng | Người phê duyệt |
> |-------------------|-----------------|
> | Dưới **5.000.000 VNĐ** | Trưởng phòng (Manager) |
> | Từ **5.000.000 - 50.000.000 VNĐ** | Giám đốc phòng ban (Director) |
> | Trên **50.000.000 VNĐ** | Tổng Giám đốc (CEO) |
>
> ## Quy trình đề xuất
> 1. Tạo phiếu đề xuất mua sắm trên hệ thống Procurement Portal
> 2. Đính kèm ít nhất 3 báo giá cho đơn hàng trên 10.000.000 VNĐ
> 3. Chờ phê duyệt theo thẩm quyền tương ứng
> 4. Phòng Mua sắm đặt hàng và theo dõi giao nhận
>
> ## Lưu ý đặc biệt
> Mua sắm thiết bị CNTT (laptop, server, phần mềm) cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất. Đơn hàng khẩn cấp có thể bỏ qua yêu cầu 3 báo giá nhưng phải có giải trình bằng văn bản.

> Nguồn: hoan_chi_dao_tao.md
> # Chính sách hoàn chi đào tạo
> > Phiên bản: 1.1 | Ngày hiệu lực: 01/07/2023 | Phòng ban: Nhân sự & Tài chính
>
> ## Điều kiện được tài trợ
> Công ty chi trả chi phí đào tạo bên ngoài (chứng chỉ, khóa học dài hạn, hội thảo) cho nhân viên có **thâm niên từ 1 năm trở lên**. Chi phí được tài trợ tối đa **30.000.000 VNĐ/khóa** và cần phê duyệt của Giám đốc phòng ban.
>
> ## Cam kết hoàn chi
> Nhân viên được tài trợ đào tạo phải **cam kết làm việc tại công ty ít nhất 1 năm** sau khi hoàn thành khóa học. Nếu nghỉ việc trước thời hạn cam kết, nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ.
>
> ## Quy trình
> 1. Nộp đơn đề xuất đào tạo kèm chương trình chi tiết
> 2. Trưởng phòng đánh giá mức độ liên quan đến công việc
> 3. Phòng Nhân sự xem xét ngân sách
> 4. Ký cam kết hoàn chi trước khi đăng ký khóa học

> Nguồn: phan_loai_du_lieu.md
> # Chính sách phân loại dữ liệu
> > Phiên bản: 1.0 | Ngày hiệu lực: 01/06/2024 | Phòng ban: CNTT & An ninh thông tin
>
> ## Bốn cấp độ phân loại
> | Cấp độ | Nhãn | Ví dụ |
> |--------|------|-------|
> | 1 | **Công khai** | Thông tin marketing, bài blog, tuyển dụng |
> | 2 | **Nội bộ** | Quy trình nội bộ, biên bản họp, danh bạ nhân viên |
> | 3 | **Bí mật** | Dữ liệu khách hàng, hợp đồng, chiến lược kinh doanh |
> | 4 | **Tối mật** | Mã nguồn core, bí mật thương mại, kế hoạch M&A |
>
> ## Quy tắc xử lý
> - **Công khai**: chia sẻ tự do, không hạn chế
> - **Nội bộ**: chỉ chia sẻ nội bộ, không ra ngoài công ty
> - **Bí mật**: mã hóa khi truyền, hạn chế quyền truy cập theo need-to-know
> - **Tối mật**: mã hóa end-to-end, truy cập cần phê duyệt Giám đốc CNTT, log tất cả truy cập
>
> ## Vi phạm
> Vi phạm chính sách phân loại dữ liệu sẽ bị xử lý kỷ luật theo Quy chế kỷ luật công ty, mức cao nhất là sa thải và chịu trách nhiệm pháp lý.

## Case Study

Câu hỏi: Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

Error Tree: Output đúng? Một phần → Context đủ? Thiếu bảng lương trong kết quả truy xuất → Query rõ? Có, hai khía cạnh → Fix module: M2/M3. Corpus đã có `bang_luong_2024.md`; ưu tiên kiểm tra candidate và khả năng phủ nhiều khía cạnh.

Nếu có thêm một giờ, ưu tiên kiểm chứng nguyên nhân của ca điểm thấp nhất, sửa module tương ứng rồi chạy lại trên toàn bộ 20 câu để tránh tối ưu riêng một câu.
