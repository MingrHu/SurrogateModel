def datasolver(txt_path,txt_savepath):
    with open(txt_path, 'r',encoding = 'utf-8') as file:
    # 获取参数文件数据信息
        lines = file.readlines() 
        res = []
        for i,line in enumerate(lines):
            arr = line.split()
            newline = f"{arr[0]}\t{arr[1]}\t{arr[2]}\t{arr[4]}\t{arr[5]}\t{arr[6]}\n"
            res.append(newline)
    with open(txt_savepath,'w',encoding = 'utf-8') as file:
        file.writelines(res)
    print(f"数据集已经制作完成，请查看文件：{txt_savepath}")

if __name__ == "__main__":

    datasolver("C:\\Users\\16969\\Desktop\\2025-07-26-21-27-12-RES.txt",
               "C:\\Users\\16969\\Desktop\\res.txt")