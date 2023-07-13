import random

def chon_menu():
    menu_ngay1 = {
        'sang': 'Menu sáng',
        'trua': ['Menu trưa 1', 'Menu trưa 2', 'Menu trưa 3', 'Menu trưa 4', 'Menu trưa 5', 'Menu trưa 6'],
        'toi': ['Menu tối 1', 'Menu tối 2', 'Menu tối 3', 'Menu tối 4', 'Menu tối 5', 'Menu tối 6', 'Menu tối 7', 'Menu tối 8', 'Menu tối 9']
    }
    
    menu_ngay2 = {
        'sang': 'Menu sáng',
        'trua': ['Menu trưa 1', 'Menu trưa 2'],
        'toi': ['Menu tối 1', 'Menu tối 2', 'Menu tối 3', 'Menu tối 4']
    }
    
    # Chọn menu ngày 1
    print("Ngày 1:")
    print("Bữa sáng:", menu_ngay1['sang'])
    print("Bữa trưa:", random.choice(menu_ngay1['trua']))
    print("Bữa tối:", random.choice(menu_ngay1['toi']))
    print()
    
    # Chọn menu ngày 2
    print("Ngày 2:")
    print("Bữa sáng:", menu_ngay2['sang'])
    print("Bữa trưa:", random.choice(menu_ngay2['trua']))
    print("Bữa tối:", random.choice(menu_ngay2['toi']))

# Gọi hàm chon_menu() để chọn menu cho hai ngày
chon_menu()





