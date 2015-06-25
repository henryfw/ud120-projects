

def add_new_feature(data_dict) :

    for name in data_dict:

        data_point = data_dict[name]


        fraction_from_poi = computeFraction( data_point["from_poi_to_this_person"], data_point["to_messages"] )
        data_dict[name]['fraction_from_poi'] = fraction_from_poi

        fraction_to_poi = computeFraction( data_point["from_this_person_to_poi"], data_point["from_messages"] )
        data_dict[name]['fraction_to_poi'] = fraction_to_poi

        fraction_to_shared_with_poi = computeFraction( data_point["shared_receipt_with_poi"], data_point["to_messages"] )
        data_dict[name]['fraction_to_shared_with_poi'] = fraction_to_shared_with_poi

    return data_dict




def computeFraction( poi_messages, all_messages ):
    all_messages = float(all_messages)
    fraction = poi_messages / all_messages if poi_messages != 'NaN' else 'NaN'
    return fraction




def normalize(data_dict, fields) :

    fields_max = {}
    fields_min = {}

    for field in fields:
        if field == 'poi' :
            continue
        fields_max[field] = None
        fields_min[field] = None
        for i in data_dict :
            value = data_dict[i][field]
            if value != 'NaN' :
                if fields_max[field] is None or value > fields_max[field] : fields_max[field] = value
                if fields_min[field] is None or value < fields_min[field] : fields_min[field] = value

    for field in fields:
        if field == 'poi' :
            continue
        for i in data_dict :
            field_range = float(fields_max[field] - fields_min[field])
            if data_dict[i][field] == 'NaN':
                new_value = 0.
            else :
                new_value = data_dict[i][field]
                if field_range > 0 :
                    new_value = ( data_dict[i][field] - fields_min[field] ) / field_range
                #print field, data_dict[i][field], new_value, field_range, fields_max[field], fields_min[field]
            data_dict[i][field + '_n'] =  new_value

    return data_dict