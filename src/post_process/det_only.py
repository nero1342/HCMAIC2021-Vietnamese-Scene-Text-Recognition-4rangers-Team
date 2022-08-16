#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8
import argparse
import importlib
import math
import sys
from collections import namedtuple
import Polygon as plg 
import rrc_evaluation_funcs
import numpy as np

def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        "IOU_CONSTRAINT": 0.5,
        "GT_SAMPLE_NAME_2_ID": "img_([0-9]+).*.txt",
        "DET_SAMPLE_NAME_2_ID": "img_([0-9]+).*.txt",
        "LTRB": False,
        "CRLF": False,  # Lines are delimited by Windows CRLF format
        # Detections must include confidence value. MAP and MAR will be calculated,
        "CONFIDENCES": False,
    }

def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    num_points = len(points)
    # resBoxes=np.empty([1,num_points],dtype='int32')
    resBoxes = np.empty([1, num_points], dtype="float32")
    for inp in range(0, num_points, 2):
        resBoxes[0, int(inp / 2)] = float(points[int(inp)])
        resBoxes[0, int(inp / 2 + num_points / 2)
                    ] = float(points[int(inp + 1)])
    pointMat = resBoxes[0].reshape([2, int(num_points / 2)]).T
    return plg.Polygon(pointMat)

def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG)

def get_intersection_over_union(pD, pG):
    try:
        return get_intersection(pD, pG) / get_union(pD, pG)
    except:
        return 0

def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    perSampleMetrics = {}

    matchedSum = 0
    det_only_matchedSum = 0
    Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

    gt = rrc_evaluation_funcs.load_zip_file(
        gtFilePath, evaluationParams["GT_SAMPLE_NAME_2_ID"])
    subm = rrc_evaluation_funcs.load_zip_file(
        submFilePath, evaluationParams["DET_SAMPLE_NAME_2_ID"], True)
    print("submission file path: ", submFilePath)
    print("gt file path: ", gtFilePath)
    numGlobalCareGt = 0
    numGlobalCareDet = 0
    det_only_numGlobalCareGt = 0
    det_only_numGlobalCareDet = 0

    for resFile in gt:
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])

        if gtFile is None:
            raise Exception("The file %s is not UTF-8" % resFile)

        recall = 0
        precision = 0
        hmean = 0
        detCorrect = 0
        detOnlyCorrect = 0
        iouMat = np.empty([1, 1])
        gtPols = []
        detPols = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCarePolsNum = []  # Array of Ground Truth Polygons' keys marked as don't Care
        det_only_gtDontCarePolsNum = []
        detDontCarePolsNum = []  # Array of Detected Polygons' matched with a don't Care GT
        det_only_detDontCarePolsNum = []
        detMatchedNums = []

        (pointsList, _, transcriptionsList,) = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(
            gtFile, evaluationParams["CRLF"], evaluationParams["LTRB"], True, False
        )

        for n in range(len(pointsList)):
            points = pointsList[n]
            
            gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)

        if resFile in subm:
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile])
            (
                pointsList,
                confidencesList,
                transcriptionsList,
            ) = rrc_evaluation_funcs.get_tl_line_values_from_file_contents_det(
                detFile,
                evaluationParams["CRLF"],
                evaluationParams["LTRB"],
                True,
                evaluationParams["CONFIDENCES"],
            )

            for n in range(len(pointsList)):
                points = pointsList[n]
                detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)
                
            if len(gtPols) > 0 and len(detPols) > 0:
                # Calculate IoU and precision matrixs
                outputShape = [len(gtPols), len(detPols)]
                iouMat = np.empty(outputShape)
                maxIoU = np.zeros([len(gtPols)]).astype(int) - 1
                
                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)
                det_only_gtRectMat = np.zeros(len(gtPols), np.int8)
                det_only_detRectMat = np.zeros(len(detPols), np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum, detNum] = get_intersection_over_union(
                            pD, pG)
                        if gtNum in gtDontCarePolsNum or detNum in detDontCarePolsNum:
                            continue 
                        if maxIoU[gtNum] == -1 or iouMat[gtNum, detNum] > iouMat[gtNum, maxIoU[gtNum]]:
                            maxIoU[gtNum] = detNum 

                cnt = 0
                # for detNum in range(len(detPols)):
                for gtNum in range(len(gtPols)):
                        # print("gt", gtTrans[gtNum].upper())
                        detNum = maxIoU[gtNum]
                        if (
                            gtRectMat[gtNum] == 0
                            and detRectMat[detNum] == 0
                        ):
                            if iouMat[gtNum, detNum] > evaluationParams["IOU_CONSTRAINT"]:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                detCorrect += 1

        numGtCare = len(gtPols) 
        numDetCare = len(detPols) 
        
        matchedSum += detCorrect
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare
        
    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    methodHmean = (
        0
        if methodRecall + methodPrecision == 0
        else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
    )
    print("Matched:" , matchedSum, "GT: ", numGlobalCareGt, "Det:", numGlobalCareDet)
    methodMetrics = r"DETECTION_ONLY_RESULTS: precision: {}, recall: {}, hmean: {}".format(
        methodPrecision, methodRecall, methodHmean
    )

    resDict = {
        "calculated": True,
        "Message": "",
        "e2e_method": methodMetrics,
        "det_only_method": methodMetrics,
        "per_sample": perSampleMetrics,
    }

    return resDict


def validate_data(gtFilePath, submFilePath, evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    gt = rrc_evaluation_funcs.load_zip_file(
        gtFilePath, evaluationParams["GT_SAMPLE_NAME_2_ID"])

    subm = rrc_evaluation_funcs.load_zip_file(
        submFilePath, evaluationParams["DET_SAMPLE_NAME_2_ID"], True)


def text_eval_main(det_file, gt_file):
    return rrc_evaluation_funcs.main_evaluation(
        None,
        det_file,
        gt_file,
        default_evaluation_params,
        validate_data,
        evaluate_method,
    )

import os, shutil
def make_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s'%(name,format), destination)
    
from argparse import ArgumentParser
parser = ArgumentParser("Evalutate AIC")
parser.add_argument("--gt", default="gt/")
parser.add_argument("--predicted", default="predicted/")

args = parser.parse_args() 

make_archive(args.predicted, 'predicted.zip')
make_archive(args.gt, 'gt.zip')

text_eval_main(
    'predicted.zip',
    'gt.zip', 
)

