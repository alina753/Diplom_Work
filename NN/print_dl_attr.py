def print_dataloader_attributes(dataloader):
    """
    https://stackoverflow.com/questions/1547466/check-if-a-parameter-is-a-python-module

    :param dataloader:
    :return:
    """
    from types import ModuleType
    import inspect

    '''for attribute in dir(dataloader):
        attribute_value = getattr(dataloader, attribute)
        print(f'{attribute=}, {type(attribute_value)=}\n')
        if isinstance(attribute_value, ModuleType) or inspect.ismodule(attribute_value) or type(
                attribute_value) is type(inspect):
            print(attribute_value)'''
    print('0')
    for idx, batch in enumerate(dataloader, 1):
        print('1')
        print('Batch index: ', idx)
        print('2')
        print('Batch size: ', batch[0].size())
        print('Batch label: ', batch[1])
        break
    print('')