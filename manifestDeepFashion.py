import os
import argparse

LEN = [6, 4, 8]


def manifest(data_path, out_path):
    eval_path = os.path.join(data_path, "Eval", "list_eval_partition.txt")
    f_eval = open(eval_path, "r")
    
    train_list = []
    valid_list = []
    test_list = []
    idx = 0
    while True:
        idx += 1
        line = f_eval.readline().strip()
        if not line:
            break
        
        if idx < 3:
            continue
        
        line = line.split()
        img_path = line[0]
        kind = line[1]
        
        if kind == "train":
            train_list.append(img_path)
        elif kind == "val":
            valid_list.append(img_path)
        elif kind == "test":
            test_list.append(img_path)
    
    f_eval.close()
    
    anno_path = os.path.join(data_path, "Anno", "list_landmarks.txt")
    f_anno = open(anno_path, "r")

    f_train = []
    f_valid = []
    f_test = []

    for id in range(3):
        f_train.append(open(os.path.join(out_path, "train_{}.txt".format(id+1)), "w"))
        f_valid.append(open(os.path.join(out_path, "valid_{}.txt".format(id+1)), "w"))
        f_test.append(open(os.path.join(out_path, "test_{}.txt".format(id+1)), "w"))

    idx = 0
    skipped = 0
    while True:
        idx += 1
        line = f_anno.readline().strip()
        if not line:
            break
        
        if idx < 3:
            continue
        
        line = line.split()
        img_path = line[0]
        clothes_type = int(line[1]) - 1
        variation_type = line[2]

        valid_img = True
        res = [os.path.join(data_path, img_path)]
        for i in range(LEN[clothes_type]):
            lv = line[i*3 + 3]
            x = line[i*3 + 4]
            y = line[i*3 + 5]
            
            res.append(x)
            res.append(y)
            if lv == "2":
                valid_img = False
        
        if valid_img == True:
            if img_path in train_list:
                f_train[clothes_type].write(','.join(res)+'\n')

            if img_path in valid_list:
                f_valid[clothes_type].write(','.join(res)+'\n')

            if img_path in test_list:
                f_test[clothes_type].write(','.join(res)+'\n')
        else:
            skipped += 1
            print("Skipping {}...".format(img_path))
            
    f_anno.close()
    for id in range(3):
        f_train[id].close()
        f_valid[id].close()
        f_test[id].close()

    print("Total skipped: {}".format(skipped))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default="data",
                        help="root directory of wav files")
    parser.add_argument('-o', '--out_path', type=str, default="data",
                        help="output path")
    args = parser.parse_args()

    manifest(args.data_path, args.out_path)
