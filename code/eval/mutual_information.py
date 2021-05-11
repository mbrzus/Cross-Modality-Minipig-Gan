import json
import os
from pathlib import Path
from joblib import Parallel, delayed
from xml.etree import ElementTree as ET


path_to_eval_script = "/home/mbrzus/tools/EvaluateSegmentation/EvaluateSegmentation_build"
eval_dir_path = "/home/mbrzus/programming/masterthesis/code/eval"

def create_eval_command(gt_arr: list, test_arr: list, i: int):
    """
    Function to create a command to use EvaluateSegmentation tool

    Input:
    gt_arr : array of ground truth label paths
    test_arr : array of ground test label paths
    i : iterator (number in array)

    Output:
    command to use EvaluateSegmentation tool on a pair of ground truth and test images and create an xml output
    Note: you can modify the command to include more metrics (look "EvaluateSegmentation --help" in terminal)
    """

    gt_label = gt_arr[i]
    test_label = test_arr[i]
    command = f"{path_to_eval_script}/EvaluateSegmentation {gt_label} {test_label} -xml '{eval_dir_path}/eval_output{i}.xml' -use DICE,HDRFDST"
    return command

if __name__ == '__main__':

    # load labels
    # for the first test I am using the labels from human data and will test the metrics on 2 exact the same labels
    # with open('../metadata/label_paths.jso', 'r') as openfile:
    #     label_json = json.load(openfile)
    # TODO: fix the way of passing files to match the new structure
    path_to_data = "/Shared/sinapse/cjohnson/all_inferrrence_removed_bad"
    t2_gt = []
    t2_gen = []
    for i in Path(path_to_data).glob('*truth*'):
        t2_gt.append(str(i))
    for i in Path(path_to_data).glob('*inferred*'):
        t2_gen.append(str(i))

    t2_gen.sort()
    t2_gt.sort()
    n = len(t2_gen)
    print(len(t2_gen))
    print(len(t2_gt))


    # Using Evaluate Segmentation tool in Parallel for speedup.
    # It will create n xml files
    # Parallel(n_jobs=32)(delayed(os.system)(create_eval_command(t2_gt, t2_gen, i)) for i in range(n))
    #
    # combined xmls into one single xml and delete the singular ones
    os.system(f"xmlmerge {eval_dir_path}/eval_output* > {eval_dir_path}/combined.xml")
    for i in Path(eval_dir_path).glob("eval*"):
        os.system(f"rm {str(i)}")

    # parse through the combined xml to get the metrics
    tree = ET.parse(f"{eval_dir_path}/combined.xml")
    root = tree.getroot()
    metrics = {'DICE': 0, 'HDRFDST': 0}
    for child in root:
        if child.tag == "metrics":
            for metric in child:
                # print(metric.tag, metric.attrib)
                metrics[metric.tag] += float(metric.attrib['value'])

    print("\nMetric averaged results")
    for metric in metrics.keys():
        print(f"{metric}: {metrics[metric] / n}")  # we have to divide by n to get averaged results


