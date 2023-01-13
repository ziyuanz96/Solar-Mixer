import numpy as np
import lightgbm as light
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import warnings
from sklearn.ensemble import IsolationForest
warnings.filterwarnings('ignore')


class AnomalyDetection(object):
    def __init__(self, dataframe, interval, x_column_name, y_column_name, granularity=50 , visualize=True, fill_na=False):
        self.interval = interval
        self.x_column_name = x_column_name
        self.y_column_name = y_column_name
        self.visualize_flag = visualize
        self.granularity = granularity
        df = dataframe[interval[0]:interval[1]]

        if fill_na:
            df = df.fillna(0)
        df.sort_values(x_column_name, inplace=True)
        self.df = df[[x_column_name, y_column_name]]
        self.num_interval = df.shape[0]//self.granularity+1
        self.main_z = np.polyfit(df[x_column_name], df[y_column_name], 1)
        self.main_p = np.poly1d(self.main_z)

    def anomaly_detection(self):
        outliers_up = []
        outliers_down = []
        for i in tqdm.tqdm(range(self.num_interval), desc='anomaly detection processing'):
            if self.df[self.x_column_name][i * self.granularity] < self.granularity:
                continue
            else:
                df_split = self.df[i * self.granularity:i * self.granularity + self.granularity]
                contamination = 0.05

            x = df_split[self.x_column_name].to_frame()
            y = df_split[self.y_column_name].to_frame()

            model = IsolationForest(n_estimators=100,
                                    max_samples='auto',
                                    contamination=contamination,
                                    max_features=1,
                                    random_state=42)
            model.fit(y)
            df_split['anomaly'] = model.fit_predict(y)
            outliers = df_split.loc[df_split['anomaly'] == -1]
            outliers['side'] = 0
            for index, row in outliers.iterrows():
                ax1 = row[self.x_column_name]
                ay2 = row[self.y_column_name]
                if ax1 * self.main_z[0] + self.main_z[1] < ay2:
                    outliers.loc[index, 'side'] = 1
                else:
                    outliers.loc[index, 'side'] = -1
            outliers_up.append(outliers.loc[outliers['side'] == 1])
            outliers_down.append(outliers.loc[outliers['side'] == -1])
        outliers_up = pd.concat(outliers_up)
        outliers_down = pd.concat(outliers_down)

        up_z = np.polyfit(outliers_up[self.x_column_name], outliers_up[self.y_column_name], 2)
        up_p = np.poly1d(up_z)

        down_z = np.polyfit(outliers_down[self.x_column_name], outliers_down[self.y_column_name], 2)
        down_p = np.poly1d(down_z)

        final_decesion = []
        for index, rows in self.df.iterrows():
            a1, a2 = rows[self.x_column_name], rows[self.y_column_name]
            if a1 ** 2 * up_z[0] + a1 * up_z[1] + up_z[2] < a2 or a1 ** 2 * down_z[0] + a1 * down_z[1] + down_z[2] > a2:
                final_decesion.append(self.df.loc[index].to_frame().T)
        final_decesion = pd.concat(final_decesion, axis=0)
        normals = pd.concat([self.df, final_decesion, final_decesion])
        normals = normals.drop_duplicates(keep=False)
        if self.visualize_flag:
            self.visualize(outliers_up, outliers_down, final_decesion, normals, up_p, down_p, './images_plot/before.eps')

        return final_decesion, normals, outliers_up, outliers_down, up_p, down_p

    def visualize(self, up_abnormal_points, down_abnormal_points, abnormal_points, normal_points, up_p, down_p, savefig_name):
        # main regression line
        plt.figure()
        bwith = 0.2
        ax = plt.gca()

        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        plt.xlabel("Radiation Global Tilted", labelpad=-40, fontsize=15)
        plt.ylabel("Active Power", labelpad=-40, fontsize=15)
        plt.yticks(size=14)
        plt.xticks(size=14)

        main_line = plt.plot(self.df[self.x_column_name], self.main_p(self.df[self.x_column_name]), "m", linewidth=bwith*4)

        upper_line = plt.plot(down_abnormal_points[self.x_column_name],
                              down_p(down_abnormal_points[self.x_column_name]), "orange", linewidth=bwith*4)

        lower_line = plt.plot(up_abnormal_points[self.x_column_name],
                              up_p(up_abnormal_points[self.x_column_name]), "orange", linewidth=bwith*4)
        # abnormal points
        s_abnormal = plt.scatter(abnormal_points[self.x_column_name],
                                 abnormal_points[self.y_column_name], color='orange', s=8)
        # Normal points
        s_normal = plt.scatter(normal_points[self.x_column_name],
                               normal_points[self.y_column_name], color='dodgerblue', s=4)
        l1 = plt.legend((s_abnormal, s_normal), ('Normal value', 'Anomaly value'),
                        loc=(0.61, 0.1), frameon=False, fontsize=12)
        l2 = plt.legend(labels=['Data regression line', 'Upper bound', 'Lower bound'],
                        loc=(0.05, 0.75), frameon=False, fontsize=12)
        plt.gca().add_artist(l1)
        plt.savefig(savefig_name, transparent=True)
        plt.show()

    def anomaly_revise(self):
        # anomaly detection
        final_decesion, normals, outliers_up, outliers_down, up_p, down_p = self.anomaly_detection()

        light_model = light.LGBMRegressor(n_estimators=20, max_depth=10)
        light_model.fit(np.array(normals[self.x_column_name]).reshape(1, -1).T,
                        np.array(normals[self.y_column_name]).reshape(1, -1).T)

        light_predict = light_model.predict(np.array(final_decesion[self.x_column_name]).reshape(1, -1).T)
        light_predict = pd.DataFrame(light_predict, columns=[self.y_column_name])

        final_decesion = final_decesion.reset_index()
        normals = normals.reset_index()
        final_decesion[self.y_column_name] = light_predict
        revised_result = pd.concat([final_decesion, normals])

        if self.visualize_flag:
            self.visualize(outliers_up, outliers_down, final_decesion, normals, up_p, down_p, './images_plot/after.eps')
        # Rearrange
        revised_result.columns = revised_result.columns.str.replace('index', 'timestamp')
        revised_result.sort_values(by='timestamp', inplace=True)
        revised_result.set_index('timestamp')
        return revised_result
