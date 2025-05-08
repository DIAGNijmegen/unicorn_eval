import numpy as np
from sklearn.metrics import roc_curve

class NoduleEvaluator:
    def __init__(self, predictions_list, ground_truth_list, diameter_list):
        """
        predictions_list:   list of np.ndarray [M_i,4] per scan (x,y,z,p)
        ground_truth_list:  list of lists of (x,y,z) per scan
        diameter_list:      any of:
                              - flat list/array of floats
                              - dict {"diameter": array_of_floats}
                              - structured array with dtype=[('diameter','O')]
                                where each record is a 1-tuple of a dict
                                {'points': [{'name':..., 'diameter':...}, ...]}
        """
        self.predictions_list  = predictions_list
        self.ground_truth_list = ground_truth_list
        self.fixedFPs          = [0.125, 0.25, 0.5, 1, 2, 4, 8]

        if isinstance(diameter_list, dict) and "diameter" in diameter_list:
            self.diameter_list = np.asarray(diameter_list["diameter"], dtype=float)

        elif isinstance(diameter_list, np.ndarray) and diameter_list.dtype.names and "diameter" in diameter_list.dtype.names:
            flat_diams = []
            for rec in diameter_list:
                cell = rec["diameter"]
                # If it's a 1-tuple of a dict, unpack it:
                info = cell[0] if isinstance(cell, tuple) else cell
                # Expecting info = {'points': [ {name,diameter}, … ]}
                for p in info.get("points", []):
                    flat_diams.append(float(p["diameter"]))
            self.diameter_list = np.array(flat_diams, dtype=float)

        else:
            self.diameter_list = np.asarray(diameter_list, dtype=float)

    def computeFROC(self, FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
        # keep only non-excluded
        gt_loc, prob_loc = [], []
        for gt, prob, exc in zip(FROCGTList, FROCProbList, excludeList):
            if not exc:
                gt_loc.append(gt)
                prob_loc.append(prob)

        ndet = sum(gt_loc)
        ntot = sum(FROCGTList)
        ncand = len(prob_loc)

        fpr, tpr, thresholds = roc_curve(gt_loc, prob_loc, pos_label=1)
        if ntot == len(FROCGTList):
            print("WARNING, this system has no false positives..")
            fps = np.zeros(len(fpr))
        else:
            fps = fpr * (ncand - ndet) / totalNumberOfImages
        sens = tpr * ndet / ntot
        return fps, sens, thresholds

    def getCPM(self, fps, sens):
        """
        For each target FP/scan in self.fixedFPs, find the sens whose
        fps is closest, then return (meanSens, [sens_at_each_target]).
        """
        fixedSens = [0.0] * len(self.fixedFPs)
        for i, target in enumerate(self.fixedFPs):
            diffPrior = max(fps)
            for j, f in enumerate(fps):
                diffCurr = abs(f - target)
                if diffCurr < diffPrior:
                    fixedSens[i] = sens[j]
                    diffPrior    = diffCurr
        meanSens = np.mean(fixedSens)
        return meanSens, fixedSens

    def compute_cpm(self):
        FROCGTList, FROCProbList, excludeList = [], [], []
        total_images = len(self.predictions_list)
        pointer = 0

        for case_idx, (preds, gts) in enumerate(zip(self.predictions_list, self.ground_truth_list)):
            # <<< ADDED: ensure preds is always 2D with shape (n_preds, 4)
            preds = np.asarray(preds).reshape(-1, 4)

            n_gts = len(gts)
            diams_case = self.diameter_list[pointer : pointer + n_gts]
            print(f"[Case {case_idx}] #GTs={n_gts},  GT coords={gts},  diameters={diams_case}")
            pointer += n_gts

            matched = np.zeros(len(preds), dtype=bool)

            # match each GT
            for j, gt in enumerate(gts):
                radius_sq = (diams_case[j] / 2.0) ** 2
                d2   = np.sum((preds[:, :3] - gt) ** 2, axis=1)
                idxs = np.where(d2 < radius_sq)[0]
                if idxs.size:
                    best = idxs[np.argmax(preds[idxs, 3])]
                    FROCGTList.append(1.0)
                    FROCProbList.append(float(preds[best, 3]))
                    excludeList.append(False)
                    matched[best] = True
                else:
                    FROCGTList.append(0.0)
                    FROCProbList.append(0.0)
                    excludeList.append(True)

            # false positives
            for k in range(len(preds)):
                if not matched[k]:
                    FROCGTList.append(0.0)
                    FROCProbList.append(float(preds[k, 3]))
                    excludeList.append(False)

        fps, sens, _ = self.computeFROC(
            FROCGTList, FROCProbList, total_images, excludeList
        )
        cpm_val, fixedSens = self.getCPM(fps, sens)

        print("\n[CPM] Sensitivities at fixed FP/scan rates:")
        for fp, s in zip(self.fixedFPs, fixedSens):
            print(f"  {fp:.3f} FP/scan → sensitivity {s:.4f}")
        print(f"[CPM] = {cpm_val:.4f}\n")
        return cpm_val

def compute_cpm(predictions_list, ground_truth_list, diameter_list):
    """
    predictions_list:   list of np.ndarray [M_i,4]
    ground_truth_list:  list of lists of (x,y,z)
    diameter_list:      as above (flat, dict, or structured array)
    """
    evaluator = NoduleEvaluator(predictions_list,
                                ground_truth_list,
                                diameter_list)
    return evaluator.compute_cpm()