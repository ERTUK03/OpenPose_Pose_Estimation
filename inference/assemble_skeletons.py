from inference.score_connections import score_connection
import torch

def assemble_skeletons(peaks_list, paf_maps, limb_sequence, image_pil, connection_threshold=0.01):
    skeletons = []

    for limb_idx, (keypoint_a_idx, keypoint_b_idx) in enumerate(limb_sequence):
        paf_x = -1*paf_maps[limb_idx,:,:,1]
        paf_y = paf_maps[limb_idx,:,:,0]
        paf_x = torch.from_numpy(paf_x)
        paf_y = torch.from_numpy(paf_y)

        for peak_a in peaks_list[keypoint_a_idx]:
            for peak_b in peaks_list[keypoint_b_idx]:
                score = score_connection(peak_a, peak_b, paf_x, paf_y, 20)
                if score > connection_threshold:
                    skeletons.append({
                        'keypoint_a': peak_a,
                        'keypoint_b': peak_b,
                        'score': score
                    })

    return skeletons
