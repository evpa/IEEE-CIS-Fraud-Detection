import pandas as pd
import numpy as np


def values_normalization(dt_df, periods, columns):
    for period in periods:
        for col in columns:
            new_col = col +'_'+ period
            dt_df[col] = dt_df[col].astype(float)  

            temp_min = dt_df.groupby([period])[col].agg(['min']).reset_index()
            temp_min.index = temp_min[period].values
            temp_min = temp_min['min'].to_dict()

            temp_max = dt_df.groupby([period])[col].agg(['max']).reset_index()
            temp_max.index = temp_max[period].values
            temp_max = temp_max['max'].to_dict()

            temp_mean = dt_df.groupby([period])[col].agg(['mean']).reset_index()
            temp_mean.index = temp_mean[period].values
            temp_mean = temp_mean['mean'].to_dict()

            temp_std = dt_df.groupby([period])[col].agg(['std']).reset_index()
            temp_std.index = temp_std[period].values
            temp_std = temp_std['std'].to_dict()

            dt_df['temp_min'] = dt_df[period].map(temp_min)
            dt_df['temp_max'] = dt_df[period].map(temp_max)
            dt_df['temp_mean'] = dt_df[period].map(temp_mean)
            dt_df['temp_std'] = dt_df[period].map(temp_std)

            dt_df[new_col+'_min_max'] = (dt_df[col]-dt_df['temp_min'])/(dt_df['temp_max']-dt_df['temp_min'])
            dt_df[new_col+'_std_score'] = (dt_df[col]-dt_df['temp_mean'])/(dt_df['temp_std'])
            del dt_df['temp_min'],dt_df['temp_max'],dt_df['temp_mean'],dt_df['temp_std']
    return dt_df


def frequency_encoding(train_df, test_df, columns, self_encoding=False):
    for col in columns:
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
        if self_encoding:
            train_df[col] = train_df[col].map(fq_encode)
            test_df[col]  = test_df[col].map(fq_encode)
        else:
            train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
            test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)
    return train_df, test_df


def timeblock_frequency_encoding(train_df, test_df, periods, columns,
                                 with_proportions=True, only_proportions=False):
    for period in periods:
        for col in columns:
            new_col = col + '_' + period
            train_df[new_col] = train_df[col].astype(str) + '_' + train_df[period].astype(str)
            test_df[new_col] = test_df[col].astype(str) + '_' + test_df[period].astype(str)

            temp_df = pd.concat([train_df[[new_col]], test_df[[new_col]]])
            fq_encode = temp_df[new_col].value_counts().to_dict()

            train_df[new_col] = train_df[new_col].map(fq_encode)
            test_df[new_col] = test_df[new_col].map(fq_encode)

            if only_proportions:
                train_df[new_col] = train_df[new_col] / train_df[period + '_total']
                test_df[new_col] = test_df[new_col] / test_df[period + '_total']

            if with_proportions:
                train_df[new_col + '_proportions'] = train_df[new_col] / train_df[period + '_total']
                test_df[new_col + '_proportions'] = test_df[new_col] / test_df[period + '_total']

    return train_df, test_df


def uid_aggregation(train_df, test_df, main_columns, uids, aggregations):
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                new_col_name = col + '_' + main_column + '_' + agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                    columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                train_df[new_col_name] = train_df[col].map(temp_df)
                test_df[new_col_name] = test_df[col].map(temp_df)
    return train_df, test_df


def uid_aggregation_and_normalization(train_df, test_df, main_columns, uids, aggregations):
    for main_column in main_columns:
        for col in uids:

            new_norm_col_name = col + '_' + main_column + '_std_norm'
            norm_cols = []

            for agg_type in aggregations:
                new_col_name = col + '_' + main_column + '_' + agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                    columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                train_df[new_col_name] = train_df[col].map(temp_df)
                test_df[new_col_name] = test_df[col].map(temp_df)
                norm_cols.append(new_col_name)

            train_df[new_norm_col_name] = (train_df[main_column] - train_df[norm_cols[0]]) / train_df[norm_cols[1]]
            test_df[new_norm_col_name] = (test_df[main_column] - test_df[norm_cols[0]]) / test_df[norm_cols[1]]

            del train_df[norm_cols[0]], train_df[norm_cols[1]]
            del test_df[norm_cols[0]], test_df[norm_cols[1]]

    return train_df, test_df


def check_cor_and_remove(train_df, test_df, i_cols, new_columns, remove=False):
    # Check correllation
    print('Correlations','#'*10)
    for col in new_columns:
        cor_cof = np.corrcoef(train_df['isFraud'], train_df[col].fillna(0))[0][1]
        print(col, cor_cof)

    if remove:
        print('#'*10)
        print('Best options:')
        best_fe_columns = []
        for main_col in i_cols:
            best_option = ''
            best_cof = 0
            for col in new_columns:
                if main_col in col:
                    cor_cof = np.corrcoef(train_df['isFraud'], train_df[col].fillna(0))[0][1]
                    cor_cof = (cor_cof**2)**0.5
                    if cor_cof>best_cof:
                        best_cof = cor_cof
                        best_option = col

            print(main_col, best_option, best_cof)
            best_fe_columns.append(best_option)

        for col in new_columns:
            if col not in best_fe_columns:
                del train_df[col], test_df[col]

    return train_df, test_df


