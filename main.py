import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import cv2
import os

def load_sam(model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    return SamPredictor(sam)


def load_yolo():
    model = YOLO("yolov8n.pt")  # Load YOLOv8 Nano model
    return model

def detect_objects(image, yolo_model, text_prompt):
    class_names = yolo_model.names  # Get class names from YOLO model
    class_ids = [i for i, name in class_names.items() if name in text_prompt.lower()]
     # Convert to uint8 if necessary

    results = yolo_model(image)
    boxes = []
    for result in results:
        for box in result.boxes:
            if box.cls in class_ids:  # Filter by class ID
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
    return boxes



def segment_objects(image, sam_predictor, boxes):
    sam_predictor.set_image(image)
    masks = []
    for box in boxes:
        mask, _, _ = sam_predictor.predict(box=np.array(box), multimask_output=False)
        masks.append(mask)
    return masks

def clean_person_mask(mask, border_thickness=2):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(mask)

    cv2.drawContours(cleaned_mask, contours, -1, color=1, thickness=cv2.FILLED)
    return cleaned_mask

def visualize_results(image, masks=None):
    if masks is not None:
        mask = masks[0][0]
        mask = clean_person_mask(mask)
        image[mask==1] = 0
        file_name = 'masked'
    else:
        file_name = 'org'
    cv2.imwrite(file_name + '.png', image)





def mp4_to_frames(video_path, output_folder, num_frames):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_frames = min(total_frames, num_frames)
    interval = max(1, total_frames // valid_frames)

    frame_count = 0
    saved_count = 0
    img_list = []
    while saved_count < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        # im = np.transpose(frame,(2,0,1))
        cv2.imwrite(output_folder+f'frame_{saved_count}.png',frame)

        saved_count += 1
        frame_count += interval
    cap.release()

    print(f"Finished extracting {saved_count} frames.")






def main():


    sam_predictor = load_sam()
    yolo_model = load_yolo()

    from sam2_train.build_sam import build_sam2_video_predictor
    config_file, ckpt_path = {
        "large": ("sam2_hiera_l.yaml", "./sam2_weights/sam2_hiera_large.pt"),
        "base_plus": ("sam2_hiera_b+.yaml", "./sam2_weights/sam2_hiera_base_plus.pt"),
        "small": ("sam2_hiera_s.yaml", "./sam2_weights/sam2_hiera_small.pt"),
        "tiny": ("sam2_hiera_t.yaml", "./sam2_weights/sam2_hiera_tiny.pt"),
    }["small"]

    sam2 = build_sam2_video_predictor(config_file=config_file, ckpt_path=ckpt_path, mode=None,
                                           apply_postprocessing=False)

    run_frame_generation = 0
    frames_folder = "./output_frames/"
    processed_frames_folder = "./processed_output_frames/"

    if run_frame_generation:
        video_path = "vv.mp4"
        num_frames = 1000  # Number of frames to extract
        mp4_to_frames(video_path, frames_folder, num_frames)
    else:
        filenames_list = os.listdir(frames_folder)
        sorted_filenames = sorted(filenames_list, key=lambda x: int(x.split('_')[1].split('.')[0]))

    inference_state = {}
    # image_array = np.stack(1, axis=0)
    # imgs_tensor =  torch.tensor(image_array, dtype=torch.uint8).unsqueeze(0)
    # inference_state['images'] =  torch.tensor(image_array, dtype=torch.uint8).unsqueeze(0)
    processed_img = []
    with torch.no_grad():

        # sam2.reset_state(inference_state)
        reverse = False
        fps = 30
        height, width, channels = (1024,1024,3)

        output_mp4 = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 format
        video_writer = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
        check = 1
        counter = -1
        id = -1
        for filename in sorted_filenames:
            id += 1
            print('ID=', id)
            counter += 1
            im_show = torch.tensor(cv2.imread(frames_folder + sorted_filenames[id])).permute(2, 0, 1).unsqueeze(0)
            inference_state = sam2.train_init_state(inference_state=inference_state,idd = id,
                                                    imgs_tensor=im_show)

            if id==0 or check:
                check = 0
                text_prompt = "person"
                im = im_show.permute(0, 2, 3, 1).squeeze(0).numpy()  # Shape: (H, W, 3)

                # Ensure NumPy array is float32 and normalized
                im = im.astype(np.uint8)
                visualize_results(im)
                boxes = detect_objects(im, yolo_model, text_prompt)



                masks = segment_objects(im, sam_predictor, boxes)
                visualize_results(im, masks)

                _, _, _ = sam2.train_add_new_bbox(
                    inference_state=inference_state,
                    frame_idx=id,
                    obj_id=1,
                    bbox=boxes,
                    clear_old_points=False,
                )

            batch_size = sam2._get_obj_num(inference_state)

            for is_cond in [False, True]:
                storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                temp_frame_inds = set()
                for obj_temp_output_dict in inference_state["temp_output_dict_per_obj"].values():
                    temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
                inference_state["consolidated_frame_inds"][storage_key].update(temp_frame_inds)

                for frame_idx in temp_frame_inds:
                    consolidated_out = sam2._consolidate_temp_output_across_obj(
                        inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                    )

                    inference_state["output_dict"][storage_key][frame_idx] = consolidated_out
                    sam2._add_output_per_object(
                        inference_state, frame_idx, consolidated_out, storage_key
                    )
                    clear_non_cond_mem = sam2.clear_non_cond_mem_around_input and (
                            sam2.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                    )
                    if clear_non_cond_mem:
                        sam2._clear_non_cond_mem_around_input(inference_state, frame_idx)

                for obj_temp_output_dict in inference_state["temp_output_dict_per_obj"].values():
                    obj_temp_output_dict[storage_key].clear()
            for frame_idx in inference_state["output_dict"]["cond_frame_outputs"]:
                inference_state["output_dict"]["non_cond_frame_outputs"].pop(frame_idx, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                for frame_idx in obj_output_dict["cond_frame_outputs"]:
                    obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

            if id == 0:
                obj_ids = inference_state["obj_ids"]
                batch_size = sam2._get_obj_num(inference_state)
                if len(inference_state["output_dict"]["cond_frame_outputs"]) == 0:
                    raise RuntimeError("No points are provided; please add points first")
                clear_non_cond_mem = sam2.clear_non_cond_mem_around_input and (
                        sam2.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )

            if id in inference_state["consolidated_frame_inds"]["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = inference_state["output_dict"][storage_key][id]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    sam2._clear_non_cond_mem_around_input(inference_state, id)
            elif id in inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = inference_state["output_dict"][storage_key][id]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                if id == 11:
                    print('fdf')
                current_out, pred_masks = sam2._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=inference_state["output_dict"],
                    frame_idx=id,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                inference_state["output_dict"][storage_key][id] = current_out

            sam2._add_output_per_object(
                inference_state, id, current_out, storage_key
            )
            inference_state["frames_already_tracked"][id] = {"reverse": reverse}

            _, video_res_masks = sam2._get_orig_video_res_output(
                inference_state, pred_masks
            )

            pred_mask = video_res_masks.squeeze().cpu().numpy()> 0.5

            pred_mask = clean_person_mask(pred_mask)



            if pred_mask.max() ==0:
                check = 1
                continue

            show_background = 0
            dem_im = im_show.squeeze().permute(1, 2, 0).numpy()
            if show_background:
                dem_im[pred_mask==1] = 0
            else:

                dem_im[pred_mask==0] = 0



            show_online=0
            if show_online:
                cv2.imshow('Image', dem_im)
                cv2.waitKey(1000)




            if not isinstance(dem_im, np.ndarray) or dem_im.shape != (height, width, channels):
                print(f"Warning: Skipping invalid frame at index {id}")
                continue

            # Ensure the image is in uint8 format
            if dem_im.dtype != np.uint8:
                dem_im = np.clip(dem_im, 0, 255).astype(np.uint8)
            #
            video_writer.write(dem_im)






            temp = 8
            if counter > temp:
                try:
                    oldest_key = min(inference_state["output_dict"]["cond_frame_outputs"].keys())
                except:
                    oldest_key = 10000
                try:
                    oldest_key_non = min(inference_state["output_dict"]["non_cond_frame_outputs"].keys())
                except:
                    oldest_key_non = 0


                inference_state['images'] = inference_state['images'][1:,:,:,:]

                counter -= 1
                if oldest_key_non > oldest_key:
                    inference_state["output_dict"]["cond_frame_outputs"].pop(oldest_key, None)

                    for idx in range(len(obj_ids)):
                        inference_state["output_dict_per_obj"][idx]["cond_frame_outputs"].pop(oldest_key, None)
                        inference_state["mask_inputs_per_obj"][idx].pop(oldest_key, None)
                        inference_state["point_inputs_per_obj"][idx].pop(oldest_key, None)

                else:
                    inference_state["output_dict"]["non_cond_frame_outputs"].pop(oldest_key_non, None)
                    for idx in range(len(obj_ids)):
                        inference_state["output_dict_per_obj"][idx]["non_cond_frame_outputs"].pop(oldest_key_non,
                                                                                                  None)
                        inference_state["mask_inputs_per_obj"][idx].pop(oldest_key_non, None)
                        inference_state["point_inputs_per_obj"][idx].pop(oldest_key_non, None)
            # if id >20:
            #     break
        video_writer.release()
        print(f"Video saved successfully: {output_mp4}")



if __name__ == "__main__":
    main()