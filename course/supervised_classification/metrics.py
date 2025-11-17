import pandas as pd
from sklearn.metrics import classification_report
from course.utils import find_project_root


def metric_report(y_test_path, y_pred_path, report_path):
    #y_test = pd.read_csv(y_test_path)
    #y_pred = pd.read_csv(y_pred_path)
    #"""Create a pandas data frame called report which contains your classifier results"""
        # make report dataframe
    #report.transpose().to_csv(report_path, index=True)
    y_test = pd.read_csv(y_test_path, header=None).squeeze()
    y_pred = pd.read_csv(y_pred_path, header=None).squeeze()

    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    report.to_csv(report_path)


def metric_report_lda():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred.csv'
    report_path = base_dir / 'vignettes' / 'supervised_classification' / 'lda.csv'
    metric_report(y_test_path, y_pred_path, report_path)


def metric_report_qda():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred.csv'
    report_path = base_dir / 'vignettes' / 'supervised_classification' / 'qda.csv'
    metric_report(y_test_path, y_pred_path, report_path)
