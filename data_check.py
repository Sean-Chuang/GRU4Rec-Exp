import lib


if __name__ == '__main__':
    data_file = "data/suumo_sess_data_eval.csv"
    batch_size = 2
    data = lib.Dataset(data_file)
    print(data.get_click_offset())
    dataloader = lib.DataLoader(data, batch_size)

    for i, (input, target, mask) in enumerate(dataloader):
    	if i < 10:
    		print(input, target, mask)
