#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
LUNA16 CPM calculator — expected input: [caseId,x,y,z,p], this calculator mimics
as closely as possible the original evaluation scripts used in the LUNA16 challenge with the CPM calculate included,
this is the metric displayed on the leaderboard.
LUNA16 Evaluation:
https://www.dropbox.com/s/wue67fg9bk5xdxt/evaluationScript.zip?dl=0

"""
from __future__ import annotations
import tempfile, csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import sklearn.metrics as skl_metrics


def read_csv(filename: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(filename, newline="", encoding="utf-8-sig") as f:
        rows.extend(csv.reader(f, delimiter=","))
    return rows


class NoduleFinding:
    def __init__(
        self,
        noduleid: int | None = None,
        coordX: float | None = None,
        coordY: float | None = None,
        coordZ: float | None = None,
        coordType: str = "World",
        CADprobability: float | None = None,
        noduleType: str | None = None,
        diameter: float | None = None,
        state: str | None = None,
        seriesInstanceUID: str | None = None,
    ):
        self.id = noduleid
        self.coordX = coordX
        self.coordY = coordY
        self.coordZ = coordZ
        self.coordType = coordType
        self.CADprobability = CADprobability
        self.noduleType = noduleType
        self.diameter_mm = diameter
        self.state = state
        self.candidateID: int | None = None
        self.seriesuid = seriesInstanceUID


seriesuid_label = "seriesuid"
coordX_label = "coordX"
coordY_label = "coordY"
coordZ_label = "coordZ"
diameter_mm_label = "diameter_mm"
CADProbability_label = "probability"
fixedFPs = [0.125, 0.25, 0.5, 1, 2, 4, 8]


def getCPM(
    fps: Sequence[float], sens: Sequence[float], fixedFPs_: Sequence[float]
) -> Tuple[float, List[float]]:
    fixedSens = [0.0] * len(fixedFPs_)
    for i, fixedFP in enumerate(fixedFPs_):
        diffPrior = max(fps)
        for j, fp in enumerate(fps):
            diffCurr = abs(fp - fixedFP)
            if diffCurr < diffPrior:
                fixedSens[i] = sens[j]
                diffPrior = diffCurr

    return np.mean(fixedSens), fixedSens


def computeFROC(
    FROCGTList: Sequence[int | float],
    FROCProbList: Sequence[float],
    totalNumberOfImages: int,
    excludeList: Sequence[bool],
):
    FROCGTList_local, FROCProbList_local = [], []
    for i in range(len(excludeList)):
        if not excludeList[i]:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])

    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)

    # ------------------------------------------------------------------ #
    # Guard: if *no* positives survive the filter, return zeros so the
    # CPM becomes 0
    # ------------------------------------------------------------------ #
    if numberOfDetectedLesions < 1:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])
    # ------------------------------------------------------------------ #

    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)

    if sum(FROCGTList) == len(FROCGTList):
        fps = np.zeros(len(fpr))
    else:
        fps = (
            fpr
            * (totalNumberOfCandidates - numberOfDetectedLesions)
            / float(totalNumberOfImages)
        )
    sens = (tpr * numberOfDetectedLesions) / float(totalNumberOfLesions)
    return fps, sens, thresholds


def getNodule(annotation: List[str], header: List[str], state: str = ""):
    n = NoduleFinding()
    n.coordX = annotation[header.index(coordX_label)]
    n.coordY = annotation[header.index(coordY_label)]
    n.coordZ = annotation[header.index(coordZ_label)]
    if diameter_mm_label in header:
        n.diameter_mm = annotation[header.index(diameter_mm_label)]
    if CADProbability_label in header:
        n.CADprobability = annotation[header.index(CADProbability_label)]
    if state:
        n.state = state
    return n


def collectNoduleAnnotations(
    annotations: List[List[str]],
    seriesUIDs: Sequence[str],
):
    allNodules: Dict[str, List[NoduleFinding]] = {}
    for seriesuid in seriesUIDs:
        nodules: List[NoduleFinding] = []
        header = annotations[0]
        for row in annotations[1:]:
            if row[header.index(seriesuid_label)] == seriesuid:
                nodules.append(getNodule(row, header, state="Included"))
        allNodules[seriesuid] = nodules
    return allNodules


def collect(
    annotations_filename: str,
    seriesuids_filename: str,
):
    annotations = read_csv(annotations_filename)
    seriesUIDs = [row[0] for row in read_csv(seriesuids_filename)]
    allNodules = collectNoduleAnnotations(annotations, seriesUIDs)
    return allNodules, seriesUIDs


def evaluateCAD_for_cpm(
    seriesUIDs: Sequence[str],
    results_filename: str,
    allNodules: Dict[str, List[NoduleFinding]],
    maxNumberOfCADMarks: int = -1,
) -> float:
    results = read_csv(results_filename)
    header = results[0]

    allCandsCAD: Dict[str, Dict[int, NoduleFinding]] = {}
    for seriesuid in seriesUIDs:
        nodules: Dict[int, NoduleFinding] = {}
        i = 0
        for res in results[1:]:
            if res[header.index(seriesuid_label)] != seriesuid:
                continue
            n = getNodule(res, header)
            n.candidateID = i
            nodules[n.candidateID] = n
            i += 1
        if 0 < maxNumberOfCADMarks < len(nodules):
            probs = sorted(
                (float(n.CADprobability) for n in nodules.values()),
                reverse=True,
            )
            probThreshold = probs[maxNumberOfCADMarks]
            nodules = {
                k: n
                for k, n in nodules.items()
                if float(n.CADprobability) > probThreshold
            }
        allCandsCAD[seriesuid] = nodules

    candTPs = candFPs = candFNs = 0
    minProbValue = -1e9

    FROCGTList: List[float] = []
    FROCProbList: List[float] = []
    FPDivisorList: List[str] = []
    excludeList: List[bool] = []

    for seriesuid in seriesUIDs:
        candidates = allCandsCAD.get(seriesuid, {})
        remaining = candidates.copy()
        noduleAnnots = allNodules.get(seriesuid, [])

        for na in noduleAnnots:
            x, y, z = map(float, (na.coordX, na.coordY, na.coordZ))
            diam = float(na.diameter_mm)
            diam = diam if diam >= 0.0 else 10.0
            rad2 = (diam / 2.0) ** 2
            matches: List[NoduleFinding] = []
            for key, cand in list(candidates.items()):
                dx = x - float(cand.coordX)
                dy = y - float(cand.coordY)
                dz = z - float(cand.coordZ)
                if dx * dx + dy * dy + dz * dz < rad2:
                    matches.append(cand)
                    remaining.pop(key, None)

            if matches:
                best = max(float(c.CADprobability) for c in matches)
                FROCGTList.append(1.0)
                FROCProbList.append(best)
                FPDivisorList.append(seriesuid)
                excludeList.append(False)
                candTPs += 1
            else:
                candFNs += 1
                FROCGTList.append(1.0)
                FROCProbList.append(minProbValue)
                FPDivisorList.append(seriesuid)
                excludeList.append(True)

        for cand in remaining.values():  # false positives
            candFPs += 1
            FROCGTList.append(0.0)
            FROCProbList.append(float(cand.CADprobability))
            FPDivisorList.append(seriesuid)
            excludeList.append(False)

    fps, sens, _ = computeFROC(FROCGTList, FROCProbList, len(seriesUIDs), excludeList)
    meanSens, _ = getCPM(fps, sens, fixedFPs)
    return meanSens


def noduleCADEvaluation_for_cpm(
    annotations_filename: str,
    seriesuids_filename: str,
    results_filename: str,
    max_number_cad_marks: int = -1,
) -> float:
    allNodules, seriesUIDs = collect(annotations_filename, seriesuids_filename)
    return evaluateCAD_for_cpm(
        seriesUIDs,
        results_filename,
        allNodules,
        maxNumberOfCADMarks=max_number_cad_marks,
    )


# ------------------------------------------------------------------ #
#  ----- helper for temporary CSV creation ------------------------- #
# ------------------------------------------------------------------ #
def _dump(rows: List[List[str]], dir_: Path, fname: str) -> str:
    path = dir_ / fname
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return str(path)


def compute_cpm(
    case_ids: Sequence[str],
    test_predictions,  # iterable of [test_id,x,y,z,p]
    test_labels: Sequence[Sequence[Tuple[float, float, float]]],
    test_extra_labels,
) -> float:
    """
    Convert run‑time tensors/lists to the CSVs required by the unchanged
    CPM evaluator and return the metric value.
    """
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # ---- seriesuids.csv ------------------------------------------------
        series_csv = _dump([[cid] for cid in case_ids], tmp_dir, "seriesuids.csv")
        # ---- annotations.csv ----------------------------------------------
        ann_rows = [
            [
                seriesuid_label,
                coordX_label,
                coordY_label,
                coordZ_label,
                diameter_mm_label,
            ]
        ]

        diam_iter = iter(test_extra_labels)
        for case_name, coords in zip(case_ids, test_labels):
            # For each coordinate triple in this case
            for x, y, z in coords:
                diam_struct = next(diam_iter)

                ann_rows.append(
                    [
                        case_name,
                        f"{x:.6f}",  # format to e.g. 6dp if you like
                        f"{y:.6f}",
                        f"{z:.6f}",
                        f"{float(diam_struct['diameter']):.6f}",
                    ]
                )
        ann_csv = _dump(ann_rows, tmp_dir, "annotations.csv")
        # ---- candidates.csv -----------------------------------------------

        cand_rows = [
            [
                seriesuid_label,
                coordX_label,
                coordY_label,
                coordZ_label,
                CADProbability_label,
            ]
        ]
        for row in test_predictions:
            test_id, x, y, z, p = row
            if test_id in case_ids:  # ignore stray predictions
                cand_rows.append([test_id, f"{x}", f"{y}", f"{z}", f"{p}"])
        cand_csv = _dump(cand_rows, tmp_dir, "candidates.csv")
        # ---- run evaluator -------------------------------------------------

        return noduleCADEvaluation_for_cpm(
            annotations_filename=ann_csv,
            seriesuids_filename=series_csv,
            results_filename=cand_csv,
            max_number_cad_marks=-1,
        )
