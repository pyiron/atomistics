def get_thermal_expansion_output(temperatures_lst, volumes_lst, output_keys):
    result_dict = {}
    if "volumes" in output_keys:
        result_dict["volumes"] = volumes_lst
    if "temperatures" in output_keys:
        result_dict["temperatures"] = temperatures_lst
    return result_dict
