task_kitti = {
    "offline":{
        0:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    },
    "16-4":{
        0:[0, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        1:[1,6,7,8]
        # 1: car
        # 6: person
        # 7: bicyclist
        # 8: motorcyclist
    },
    "16-1":{
        0:[0, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        1:[8]
        # 8: motorcyclist
    },
    "16-3":{
        0:[0, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        1:[1,6,7]
        # 1: car
        # 6: person
        # 7: bicyclist
    },
    "17-3":{
        0:[0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        1:[1,6,7]
        # 1: car
        # 6: person
        # 7: bicyclist
    },
}

# def get_labels(task_name, task_step):

def get_task_labels(name, step):
    task_dict = task_kitti[name]
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    labels = list(task_dict[step])
    labels_old = [label for s in range(step) for label in task_dict[s]]
    return labels, labels_old


def get_per_task_classes(name, step):
    task_dict = task_kitti[name]
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    classes = [len(task_dict[s]) for s in range(step+1)]
    return classes
