
def tracking():
    dataloader = MOTDataloader(os.path.join(base_path, SEQUENCE))
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS, bbox_amplification=bbox_amp)
    sequence_length = dataloader.get_seuqence_length()
    frame = dataloader.get_current_frame()

    tracker = Tracker(frame_width=frame.shape[1], frame_height=frame.shape[0])
    out = cv.VideoWriter(
        'project.mp4', cv.VideoWriter_fourcc(*'MP4V'), 25, (frame.shape[1], frame.shape[0]))

    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        # Get the set of current detections
        # dataloader.get_current_gt_bbox_scale()
        bbox_list = dataloader.get_current_det_bbox_scale()
        cv_tools.set_active_frame(frame)
        frame_list = cv_tools.blur_image_list_except_bbox(bbox_list)
        # Calculates the feature representation of each detection
        feature_list = feature_extraction.get_feature_description(
            frame_list, bbox_list)

        tracker.update_tracks(feature_list, bbox_list)
        frame = tracker.draw_active_tracks_on_frame(frame)
        # Get integration?
        tracker.add_existing_tracks_to_df(dataloader.get_current_frame_id())
        out.write(frame)
        dataloader.next_frame()
    # save in
    path = os.path.join(tracker_base_path,
                        str(DIMENSIONS[0]), 'data')
    try:
        os.makedirs(path, exist_ok=False)
    except:
        pass
    out_file = os.path.join(path, SEQUENCE + '.txt')
    tracker.save_track_file(out_file)
    out.release()



def tracking_combined():
    dataloader = MOTDataloader(os.path.join(base_path, SEQUENCE))
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS, bbox_amplification=bbox_amp)
    sequence_length = dataloader.get_seuqence_length()
    frame = dataloader.get_current_frame()

    tracker = Tracker(frame_width=frame.shape[1], frame_height=frame.shape[0])
    out = cv.VideoWriter(
        'project.mp4', cv.VideoWriter_fourcc(*'MP4V'), 30, (frame.shape[1], frame.shape[0]))

    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        # Get the set of current detections
        # dataloader.get_current_gt_bbox_scale()
        bbox_list = dataloader.get_current_det_bbox_scale()
        cv_tools.set_active_frame(frame)
        bboxes = cv_tools.extract_bbox_from_list(bbox_list)
        frames = cv_tools.blur_image_list_except_bbox(bbox_list)

        # Calculates the feature representation of each detection
        frame_feature_list, bbox_feature_list = feature_extraction.get_feature_description_combined(
            frames=frames, bbox_list=bbox_list, bboxes=bboxes)
        # Uses a collected method for global association. RE-ID with only bbox...
        frame_feature_list = feature_extraction.append_features(
            frame_feature_list, bbox_feature_list)
        tracker.update_tracks_combined(
            frame_feature_list, bbox_feature_list, bbox_list)
        frame = tracker.draw_active_tracks_on_frame(frame)
        # Get integration?
        tracker.add_existing_tracks_to_df(dataloader.get_current_frame_id())
        out.write(frame)
        dataloader.next_frame()
    # save in
    path = os.path.join(tracker_base_path,
                        str(DIMENSIONS[0]), 'data')
    try:
        os.makedirs(path, exist_ok=False)
    except:
        pass
    out_file = os.path.join(path, SEQUENCE + '.txt')
    tracker.save_track_file(out_file)
    out.release()



# Deprecated functions
def blurring_similarity(kernel):
    dataloader = MOTDataloader(SEQUENCE)
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(DIMENSIONS)

    similarity_vector = []
    sequence_length = dataloader.get_seuqence_length()
    kernel_size = (kernel, kernel)
    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        cv_tools.set_active_frame(frame)

        blur_frame = cv_tools.gaussian_blur(
            cv_tools.get_current_frame(), kernel_size)
        save_image_file('test35.png', blur_frame)
        features = feature_extraction.extract_features(
            blur_frame)
        # Perform similarity between ground truths
        if "prev_feature" in locals():
            similarity_vector.append(cosine_similarity(
                features[0], prev_feature[0]))

        prev_feature = features
        dataloader.next_frame()

    save_similarity_vector(
        similarity_vector, "sim_vec_blur_" + str(kernel) + ".pkl")


def visualize_blur(path):
    """ Plots the mean similarity accross sequence."""
    ran = [i for i in range(1, 48, 2)]
    temp = []
    print(ran)
    for i in ran:
        temp.append(np.mean(load_similarity_vector(os.path.join(path,
                                                                'sim_vec_blur_' + str(i) + '.pkl'))))
    plt.plot(ran, temp)
    plt.savefig('blur_effect.png')





def visualize_change():
    """ Plots the area vs. change"""
    sim_vec_gt = np.asarray(load_similarity_vector(
        'sim_' + str(DIMENSIONS[0]) + '_gt.pkl'))
    norm = np.asarray(load_similarity_vector("norm.pkl"))
    area = np.asarray(load_similarity_vector("area.pkl"))

    sim_vec_not_gt = np.asarray(load_similarity_vector(
        'sim_max_not_gt_' + str(DIMENSIONS[0]) + '.pkl'))
    norm_max = np.asarray(load_similarity_vector("norm_max.pkl"))
    area_max = np.asarray(load_similarity_vector("area_max.pkl"))

    plt.scatter(norm, sim_vec_gt, s=0.1, alpha=0.5)
    plt.scatter(norm_max, sim_vec_not_gt, s=0.1, c='g', alpha=0.5)
    plt.savefig('norm.png')
    plt.clf()

    plt.scatter(area, sim_vec_gt, s=0.1, alpha=0.5)
    plt.scatter(area_max, sim_vec_not_gt, s=0.1, c='g', alpha=0.5)
    plt.savefig('area.png')
    plt.clf()



def extract_similarity_not_gt():
    """ Extracts the max similarity and avg similarity between not GT"""
    dataloader = MOTDataloader(SEQUENCE)
    cv_tools = CVTools(DIMENSIONS)
    similarity_vector_avg = []
    similarity_vector_max = []
    sequence_length = dataloader.get_seuqence_length()
    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
        cv_tools.set_active_frame(frame)

        bboxes = cv_tools.extract_bbox_from_list(bbox_list)
        frame_list = cv_tools.black_image_list_except_bbox(bbox_list)
        detection_array = feature_extraction.get_detection_features(
            frame_list, bbox_list=bbox_list)
        features = [description() for description in detection_array]

        # Perform similarity between ground truths
        if "prev_id_list" in locals():
            for index, id in enumerate(id_list):
                temp_similarity = []
                temp_norm = []
                temp_area = []
                for prev_index, prev_id in enumerate(prev_id_list):
                    if prev_id != id:
                        temp_similarity.append(metric_to_extract(
                            features[index], prev_feature_list[prev_index]))
                # Average
                if len(temp_similarity) != 0:
                    avg_sim = sum(temp_similarity) / len(temp_similarity)
                    similarity_vector_avg.append(avg_sim)
                    # Max
                    min_value = max(temp_similarity)
                    similarity_vector_max.append(min_value)
        prev_id_list = id_list
        prev_feature_list = features
        dataloader.next_frame()

    save_similarity_vector(similarity_vector_max,
                           "sim_max_not_gt_" + str(DIMENSIONS[0]) + ".pkl")
    save_similarity_vector(similarity_vector_avg,
                           "sim_avg_not_gt_" + str(DIMENSIONS[0]) + ".pkl")


######### DEPRECATED ############
def extract_similarity_gt_location_different_roi():
    """ Extracts similarity between GT locations with a different bbox/roi on previous """
    dataloader = MOTDataloader(SEQUENCE)
    cv_tools = CVTools(DIMENSIONS)

    similarity_vector = []
    sequence_length = dataloader.get_seuqence_length()
    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
        cv_tools.set_active_frame(frame)

        # Feature extraction
        frame_list_gt = cv_tools.blur_image_list_except_bbox(bbox_list)
        features_gt = feature_extraction.get_feature_description(frame_list_gt)

        # Feature extraction other bbox
        new_bbox_list = bbox_list[1:] + [bbox_list[0]]
        frame_list_other_bbox = cv_tools.blur_image_list_except_bbox_newbb(
            bbox_list, new_bbox_list)
        features_gt_other = feature_extraction.get_feature_description(
            frame_list_other_bbox)

        # Perform similarity between ground truths
        if "prev_id_list" in locals():
            for index, id in enumerate(id_list):
                if id in prev_id_list:
                    prev_index = prev_id_list.index(id)
                    similarity_vector.append(metric_to_extract(
                        features_gt[index], prev_feature_list[prev_index]))
        prev_id_list = id_list
        prev_feature_list = features_gt_other
        dataloader.next_frame()

    save_similarity_vector(
        similarity_vector, "sim_gt_" + str(DIMENSIONS[0]) + "_new_roi.pkl")
