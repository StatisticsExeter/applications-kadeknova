import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from course.utils import find_project_root


def metric_report(y_test_path, y_pred_path, report_path):
    y_test = pd.read_csv(y_test_path)
    y_pred = pd.read_csv(y_pred_path)
    """Create a pandas data frame called report which contains your classifier results"""
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
    report.transpose().to_csv(report_path, index=True)
#    y_test = pd.read_csv(y_test_path, header=None).squeeze()
#    y_pred = pd.read_csv(y_pred_path, header=None).squeeze()
#
#    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
#    report.to_csv(report_path)


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


def create_confusion_matrix(y_test_path, y_pred_path, outpath):
    y_test = pd.read_csv(y_test_path)
    y_pred = pd.read_csv(y_pred_path)

    cm = confusion_matrix(y_test, y_pred, labels=["Pre-30s", "Post-30s"])

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Pre-30s", "Actual Post-30s"],
        columns=["Pred Pre-30s", "Pred Post-30s"]
    )

    cm_df.to_csv(outpath)


def confusion_matrix_lda():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred.csv'
    outpath = base_dir / 'vignettes' / 'supervised_classification' / 'confusion_matrix_lda.csv'
    create_confusion_matrix(y_test_path, y_pred_path, outpath)


def confusion_matrix_qda():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred.csv'
    outpath = base_dir / 'vignettes' / 'supervised_classification' / 'confusion_matrix_qda.csv'
    create_confusion_matrix(y_test_path, y_pred_path, outpath)
