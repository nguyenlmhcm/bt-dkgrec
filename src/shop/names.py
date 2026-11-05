"""Illustrative Vietnamese shop vocabulary.

RetailRocket publishes anonymised ids only, so product names here are invented for
the demo and must always be shown as illustrative. The mapping is deterministic and
keeps the real category structure: items sharing a category share a product type.
"""

from __future__ import annotations

#: (department, icon, label colour index, product types, variants). Listed from the
#: largest real root category down; the assignment to a real root is arbitrary.
DEPARTMENTS: list[tuple[str, str, list[str], list[str]]] = [
    ("Nhà bếp", "cooking-pot",
     ["Chảo chống dính", "Nồi inox", "Bộ dao bếp", "Thớt gỗ", "Bình giữ nhiệt", "Hộp đựng thực phẩm", "Ấm đun nước",
      "Nồi cơm điện", "Bộ chén sứ", "Khay nướng", "Muỗng đũa", "Máy xay sinh tố", "Bếp từ đơn", "Rổ inox"],
     ["loại dày", "dùng hằng ngày", "bản gia đình", "màu kem", "màu xám", "cỡ nhỏ", "cỡ lớn", "mẫu mới"]),
    ("Thời trang", "shirt",
     ["Áo thun cotton", "Sơ mi vải lanh", "Quần tây", "Quần jean", "Váy liền", "Áo khoác gió", "Áo len mỏng", "Đầm suông",
      "Chân váy", "Quần short", "Áo polo", "Khăn choàng", "Giày vải", "Dép quai ngang"],
     ["dáng rộng", "dáng gọn", "màu be", "màu xanh rêu", "màu than", "màu trắng", "trơn", "mẫu mới"]),
    ("Điện thoại và phụ kiện", "smartphone",
     ["Ốp lưng", "Cáp sạc", "Sạc dự phòng", "Kính cường lực", "Giá đỡ điện thoại", "Tai nghe có dây", "Củ sạc nhanh",
      "Thẻ nhớ", "Gậy chụp ảnh", "Đế sạc không dây", "Bao da", "Pin thay thế"],
     ["màu đen", "màu trắng", "loại bền", "loại mỏng", "bản tiêu chuẩn", "bản nâng cấp", "mẫu mới", "cỡ gọn"]),
    ("Nội thất", "armchair",
     ["Bàn làm việc", "Ghế xoay", "Kệ sách", "Tủ quần áo", "Giường gỗ", "Bàn trà", "Ghế đôn", "Tủ đầu giường",
      "Kệ giày", "Đèn sàn", "Rèm cửa", "Thảm trải sàn", "Gương đứng", "Sofa đơn"],
     ["màu trắng", "màu gỗ tự nhiên", "màu óc chó", "cỡ nhỏ", "cỡ lớn", "kiểu tối giản", "kiểu cổ điển", "mẫu mới"]),
    ("Làm đẹp", "sparkles",
     ["Sữa rửa mặt", "Kem chống nắng", "Son lì", "Dầu gội", "Mặt nạ giấy", "Nước tẩy trang", "Kem dưỡng ẩm", "Serum",
      "Sữa tắm", "Lược chải tóc", "Bông tẩy trang", "Nước hoa mini"],
     ["dịu nhẹ", "hương trà", "hương cam", "không mùi", "cỡ du lịch", "cỡ lớn", "mẫu mới", "bản tiêu chuẩn"]),
    ("Mẹ và bé", "baby",
     ["Tã dán", "Bình sữa", "Khăn sữa", "Xe đẩy", "Ghế ăn dặm", "Đồ chơi gỗ", "Bộ quần áo sơ sinh", "Yếm ăn",
      "Sữa bột", "Nôi điện", "Thảm chơi", "Núm ti"],
     ["cho bé 0-6 tháng", "cho bé 6-12 tháng", "cho bé 1-3 tuổi", "loại nhẹ", "màu pastel", "màu trắng", "mẫu mới", "bản tiêu chuẩn"]),
    ("Thể thao", "dumbbell",
     ["Thảm tập", "Tạ tay", "Dây kháng lực", "Giày chạy", "Bình nước thể thao", "Vợt cầu lông", "Bóng đá", "Xe đạp tập",
      "Găng tay tập", "Áo tập", "Dây nhảy", "Ba lô leo núi"],
     ["cỡ tiêu chuẩn", "loại nhẹ", "màu cam", "màu xanh", "màu đen", "cho người mới", "bản nâng cao", "mẫu mới"]),
    ("Sách và văn phòng phẩm", "book-open",
     ["Sổ tay", "Bút bi", "Sách kỹ năng", "Tiểu thuyết", "Bút dạ quang", "Giấy in A4", "Kẹp tài liệu", "Sách thiếu nhi",
      "Băng keo", "Bìa còng", "Máy tính cầm tay", "Lịch để bàn"],
     ["bản tiêu chuẩn", "bản tái bản", "bộ 3 món", "bộ 10 món", "màu xanh", "màu đen", "cỡ nhỏ", "mẫu mới"]),
    ("Dụng cụ sửa chữa", "wrench",
     ["Máy khoan", "Bộ tua vít", "Cờ lê", "Thước cuộn", "Đèn pin", "Thang nhôm", "Keo dán đa năng", "Máy mài cầm tay",
      "Kìm cắt", "Sơn xịt", "Băng dính điện", "Hộp đồ nghề"],
     ["loại nhỏ", "loại chuyên dụng", "bản tiêu chuẩn", "chống gỉ", "cầm tay", "dùng cho gia đình", "mẫu mới", "loại bền"]),
    ("Xe và phụ kiện", "car",
     ["Mũ bảo hiểm", "Thảm lót xe", "Dầu nhớt", "Áo mưa", "Giá đỡ trên xe", "Bọc ghế", "Nước rửa xe", "Găng tay đi xe",
      "Camera hành trình", "Khăn lau xe"],
     ["loại bền", "màu đen", "màu xám", "cỡ phổ thông", "chống nước", "cho xe máy", "cho ô tô", "mẫu mới"]),
    ("Thủ công và quà tặng", "pen-tool",
     ["Hộp quà", "Giấy gói", "Nến thơm", "Khung ảnh", "Bộ màu nước", "Len đan", "Thiệp chúc mừng", "Đất sét nặn"],
     ["làm tay", "bộ 3 món", "cỡ nhỏ", "cỡ vừa", "màu nâu", "tông trầm", "tông sáng", "mẫu mới"]),
    ("Thú cưng", "dog",
     ["Hạt cho chó", "Pate cho mèo", "Cát vệ sinh", "Dây dắt", "Đệm nằm", "Đồ chơi gặm", "Bát ăn đôi", "Sữa tắm thú cưng"],
     ["cỡ nhỏ", "cỡ vừa", "cỡ lớn", "cho chó nhỏ", "cho mèo con", "loại mềm", "mẫu mới", "bản tiêu chuẩn"]),
    ("Thực phẩm", "leaf",
     ["Cà phê rang xay", "Trà ô long", "Hạt điều", "Mật ong", "Bánh quy", "Gạo thơm", "Nước mắm", "Hạt chia"],
     ["gói nhỏ", "gói lớn", "đóng hộp", "vị nguyên bản", "loại đặc biệt", "mùa mới", "hộp quà", "bản tiêu chuẩn"]),
    ("Trò chơi và giải trí", "gamepad-2",
     ["Tay cầm chơi game", "Bộ cờ", "Xếp hình", "Boardgame gia đình", "Ô tô điều khiển", "Khối lắp ghép"],
     ["từ 6 tuổi", "từ 12 tuổi", "cho gia đình", "cỡ bỏ túi", "bản mở rộng", "bản tiêu chuẩn", "mẫu mới", "màu xanh"]),
    ("Du lịch", "luggage",
     ["Vali kéo", "Túi du lịch", "Gối cổ", "Túi đựng mỹ phẩm", "Ổ cắm du lịch", "Khóa số"],
     ["gấp gọn", "chống nước", "màu xám", "màu xanh than", "loại nhẹ", "cỡ nhỏ", "cỡ lớn", "mẫu mới"]),
    ("Âm thanh", "headphones",
     ["Tai nghe không dây", "Loa bluetooth", "Micro thu âm", "Dây tín hiệu"],
     ["màu trắng", "màu đen", "cỡ bỏ túi", "kết nối đôi", "USB-C", "bản tiêu chuẩn", "bản nâng cấp", "mẫu mới"]),
    ("Trang sức", "gem",
     ["Dây chuyền bạc", "Nhẫn", "Bông tai", "Vòng tay"],
     ["bạc 925", "mạ vàng", "đính đá", "tối giản", "khắc tên", "cỡ nhỏ", "cỡ vừa", "mẫu mới"]),
    ("Điện gia dụng", "refrigerator",
     ["Quạt đứng", "Máy lọc không khí", "Bàn ủi hơi nước"],
     ["có hẹn giờ", "tiết kiệm điện", "màu trắng", "màu xám", "loại nhỏ", "loại lớn", "có điều khiển", "mẫu mới"]),
    ("Điện tử", "tv",
     ["Ổ cắm thông minh", "Chuột không dây", "Bàn phím cơ"],
     ["kết nối wifi", "pin sạc", "màu đen", "màu trắng", "cỡ gọn", "bản tiêu chuẩn", "bản nâng cấp", "mẫu mới"]),
    ("Nhà vườn", "flower-2",
     ["Chậu cây", "Hạt giống"],
     ["dễ trồng", "cỡ nhỏ", "cỡ vừa", "để bàn", "treo ban công", "gói nhỏ", "mẫu mới", "bản tiêu chuẩn"]),
    ("Đồ dùng khác", "package",
     ["Móc treo"],
     ["bộ 10 cái", "dán tường", "inox", "nhựa", "chịu lực", "cỡ nhỏ", "cỡ lớn", "màu trắng"]),
]

#: Label colourways: the colour named in a product's title and the tone of its label.
COLOURS = ["vàng nghệ", "đỏ gạch", "xanh than", "xanh ngọc nhạt", "hồng đất", "xanh rêu", "vàng chanh", "xanh da trời"]

#: Departments whose goods are not sold by colour; the tone is still used on the label.
NO_COLOUR = {"Thực phẩm", "Sách và văn phòng phẩm", "Thú cưng", "Nhà vườn"}

#: Used for items whose category is unknown at T_train (about 9% of the catalogue).
UNCATEGORISED = "Hàng chưa phân loại"

#: Illustrative customer first names.
PERSONAS = ["Minh Anh", "Thu Hà", "Quốc Bảo", "Ngọc Lan", "Gia Huy", "Thanh Tâm", "Hoàng Nam", "Khánh Linh",
            "Đức Anh", "Phương Vy", "Tuấn Kiệt", "Bảo Châu", "Hải Yến", "Trung Hiếu", "Mai Chi", "Nhật Minh"]
