import pandas as pd


class data:
    def __init__(self, path):
        self.path = path
        self.header = 19
        self.columns = ["TIME", "CH1", "CH2", "CH3", "CH4"]
        self.read_range = []
        self.ifprint = True
        self.return_data = []
        self.normalized_data = []

    def read(self):
        with open(self.path, "r") as file:
            csv_data = pd.read_csv(
                file,
                header=self.header,
            )
        self.return_data = []
        for column in self.columns:
            try:
                # 默认全部读取
                if self.read_range == []:
                    self.return_data.append(csv_data.loc[:, column])
                else:
                    self.return_data.append(
                        csv_data.loc[self.read_range[0] : self.read_range[1], column]
                    )
            except:
                if self.ifprint:
                    # 空数据占位，保持原始通道编号
                    self.return_data.append([])
                    print("In data.read(), channel " + column + " is blank.")
        return self.return_data

    def normalize(self):
        self.normalized_data.append(self.return_data[0])
        for i in range(1, len(self.return_data)):
            try:
                self.normalized_data.append(
                    self.return_data[i] / max(self.return_data[i])
                )
            except:
                self.normalized_data.append([])
                if self.ifprint:
                    print("In data.normalize(), channel " + str(i) + " error.")

        return self.normalized_data


if __name__ == "__main__":
    dir = "D:/001_zerlingx/notes/literature/HC/007_experiments/2023-07 一号阴极测试/2023-08-30 点火与单探针测试/data/RAW"
    path = "/tek0011ALL.csv"
    data = data(dir + path)
    res = data.read()
    res = data.normalize()
