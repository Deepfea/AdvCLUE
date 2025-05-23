from cmath import sqrt
def get_rem(dsr_value_list, dfe_value_list):
    sum_value = 0
    for num in range(len(dsr_value_list)):
        temp_value = sqrt(dsr_value_list[num] / (1 + dfe_value_list[num]))
        sum_value += temp_value
    avg_value = sum_value / float(len(dsr_value_list))
    return avg_value

if __name__ == '__main__':
    pass