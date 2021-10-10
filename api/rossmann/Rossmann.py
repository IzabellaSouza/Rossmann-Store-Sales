import math
import pickle
import datetime
import inflection
import pandas as pd
import numpy  as np



class Rossmann (object):
    def __init__(self):
        state = 1
        self.home_path = 'C:\Users\Usuario\Projetos\Rossmann-Store-Sales/'
        self.competition_distance_scaler   = pickle.load(open(self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler        = pickle.load(open(self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open(self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.store_type_scaler             = pickle.load(open(self.home_path + 'parameter/store_type_scaler.pkl', 'rb'))

    def data_cleaning(self, dataset1):

        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']

        def snakecase(x): return inflection.underscore(x)

        cols_new = list(map(snakecase, cols_old))
        
        # rename
        dataset1.columns = cols_new
        
        # Dta Types
        dataset1['date'] = pd.to_datetime(dataset1['date'])

        # Fillout N/A
        # competition_distance
        dataset1['competition_distance'] = dataset1['competition_distance'].apply(lambda x: 200000 if math.isnan(x) else x)

        # competition_open_since_month
        dataset1['competition_open_since_month'] = dataset1.apply(lambda x: x['date'].month if math.isnan(
            x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

        # competition_open_since_year
        dataset1['competition_open_since_year'] = dataset1.apply(lambda x: x['date'].year if math.isnan(
            x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)

        # promo2_since_week
        dataset1['promo2_since_week'] = dataset1.apply(lambda x: x['date'].week if math.isnan(
            x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        # promo2_since_year
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(
            x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        # promo_interval
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        dataset1['promo_interval'].fillna(0, inplace=True)
        dataset1['month_map'] = df1['date'].dt.month.map(month_map)

        # eh promocao quando a promo2 coincide com a date
        dataset1['is_promo'] = dataset1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        # 1.6. Change Types
        dataset1['competition_open_since_month'] = dataset1['competition_open_since_month'].astype(int)
        dataset1['competition_open_since_year']  = dataset1['competition_open_since_year'].astype(int)
        dataset1['promo2_since_week']            = dataset1['promo2_since_week'].astype(int)
        dataset1['promo2_since_year']            = dataset1['promo2_since_year'].astype(int)

        return dataset1

    def feature_engineering(self, dataset2):

        # year
        dataset2['year'] = dataset2['date'].dt.year

        # month
        dataset2['month'] = dataset2['date'].dt.month
        
        # day
        dataset2['day'] = dataset2['date'].dt.day
        
        # week of year
        dataset2['week_of_year'] = dataset2['date'].dt.weekofyear
        
        # year week
        dataset2['year_week'] = dataset2['date'].dt.strftime('%Y-%W')

        # competition since
        dataset2['competition_since'] = dataset2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)
        
        # competition time month
        dataset2['competition_time_month'] = ((dataset2['date'] - dataset2['competition_since']) / 30).apply(lambda x: x.days).astype(int)

        # promo since
        dataset2['promo_since'] = dataset2['promo2_since_year'].astype(str) + '-'+ dataset2['promo2_since_week'].astype(str)
        dataset2['promo_since'] = dataset2['promo_since'].apply(lambda x: datetime.datetime.strptime(x+'-1', '%Y-%W-%w') - datetime.timedelta(days=7))
       
        # promo time week
        dataset2['promo_time_week'] = ((dataset2['date'] - dataset2['promo_since'])/7).apply(lambda x: x.days).astype(int)

        # assortment
        dataset2['assortment'] = dataset2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')
       
        # state holiday
        dataset2['state_holiday'] = dataset2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # Variabl filtering
        # Line filtering
        dataset2 = dataset2[dataset2['open'] != 0]

        # Delet restrictions
        cols_drop = ['open', 'promo_interval', 'month_map']
        dataset2 = dataset2.drop(cols_drop, axis=1)

        return dataset2

    def data_preparation(self, dataset5):

        ## Rescaling

        # Apply RobustScaler
        dataset5['competition_distance'] = self.competition_distance_scaler.fit_transform(dataset5[['competition_distance']].values)
        
        # competition time month
        dataset5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(dataset5[['competition_time_month']].values)

        # promotimeweek
        dataset5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(dataset5[['promo_time_week']].values)

        # year
        dataset5['year'] = self.year_scaler.fit_transform(df5[['year']].values)

        ## Encoding

        # One hot encoding - State holiday
        dataset5 = pd.get_dummies(dataset5, prefix=['state_holiday'], columns=['state_holiday'])

        # Label Encoding - Store type
        dataset5['store_type'] = self.store_type_scaler.fit_transform(dataset5['store_type'])

        # Assortment - Ordinal Encoding
        assortment_dict = {'basic': 1, 'extended': 2, 'extra': 3}
        dataset5['assortment'] = dataset5['assortment'].map(assortment_dict)

        # Nature Transformation
        # month
        dataset5['month_sin'] = dataset5['month'].apply(lambda x: np.sin(x * (2 * np.pi/12)))
        dataset5['month_cos'] = dataset5['month'].apply(lambda x: np.cos(x * (2 * np.pi/12)))

        # day
        dataset5['day_sin'] = dataset5['day'].apply(lambda x: np.sin(x * (2 * np.pi/30)))
        dataset5['day_cos'] = dataset5['day'].apply(lambda x: np.cos(x * (2 * np.pi/30)))

        # week of year
        dataset5['week_of_year_sin'] = dataset5['week_of_year'].apply(lambda x: np.sin(x * (2 * np.pi/52)))
        dataset5['week_of_year_cos'] = dataset5['week_of_year'].apply(lambda x: np.cos(x * (2 * np.pi/52)))

        # day of week
        dataset5['day_of_week_sin'] = dataset5['day_of_week'].apply(lambda x: np.sin(x * (2 * np.pi/7)))
        dataset5['day_of_week_cos'] = dataset5['day_of_week'].apply(lambda x: np.cos(x * (2 * np.pi/7)))

        cols_selected = [

            'store',
            'promo',
            'store_type',
            'assortment',
            'competition_distance',
            'competition_open_since_month',
            'competition_open_since_year',
            'promo2',
            'promo2_since_week',
            'promo2_since_year',
            'competition_time_month',
            'promo_time_week',
            'month_sin',
            'month_cos',
            'day_sin',
            'day_cos',
            'week_of_year_sin',
            'week_of_year_cos',
            'day_of_week_sin',
            'day_of_week_cos'

        ]

        return dataset5[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # get prediction
        pred = model.predict(test_data)

        # join
        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')