import pandas as pd
import time

class GenClsDataset():
    """
    生成数据
    """

    def __init__(self):
        self.df = pd.read_csv('cls_md5.csv')
        pass

    def read_csv(self, csv_path):
        """
        遍历每一个含有landmark的数据，为每一个数据增加标签信息
        :return:
        """
        name_list = []
        code_list = []
        landmark_list = []
        same_name_list = []
        cls_list = []
        df = pd.read_csv(csv_path)
        length = df.shape[0]

        for idx,row in df.iterrows():
            start = time.time()
            print(idx, '/', length)

            name = row[1]
            code = row[2]
            landmark = row[3]
            same_name, cls = self.search_same(target_code=code)
            if same_name =='':
                continue
            name_list.append(name)
            code_list.append(code)
            landmark_list.append(landmark)
            same_name_list.append(same_name)
            cls_list.append(cls)
            end = time.time()
            print(end - start)
            print(same_name,cls)
        data = {'ImgName': name_list, 'ImgMD5': code_list,
                'landmark': landmark_list,'same_name':same_name_list,'cls':cls_list}
        df = pd.DataFrame(data)
        df.to_csv('landmark_md5_cls_clear.csv')
    def search_same(self, target_code):

        df = self.df
        find_name = []
        length = df.shape[0]
        cls_list = []

        result = df.loc[df['ImgMD5']==target_code]

        find_name=list(result['ImgName'])
        for name in find_name:
            cls_list.append(str((name.split('_')[1], name.split('_')[2])))
        #
        # for idx,row in df.iterrows():
        #
        #     name = row[1]
        #     search_code = row[2]
        #     if search_code == target_code:
        #         find_name.append(name)
        #         cls_list.append(str((name.split('_')[1], name.split('_')[2])))

        if len(find_name) == 0:
            return '', ''
        else:
            return str(find_name), str(cls_list)

    def parse_landmark(self, landmark):
        landmark = landmark.replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("\'", '')
        landmark = landmark.split(',')

        landmark_out = []
        for i in range(12):
            # print(landmark[2*i])
            # print(landmark[2 * i + 1])
            landmark_out.append(int(landmark[2 * i]))
            landmark_out.append(int(landmark[2 * i + 1]))
        return landmark_out

if __name__ == '__main__':

    GenClsDataset().read_csv('landmark_md5.csv')