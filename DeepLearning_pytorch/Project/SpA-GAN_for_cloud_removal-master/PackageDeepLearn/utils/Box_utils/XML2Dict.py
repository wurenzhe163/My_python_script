"""
Detail: 解析XML-->dict
Time:2022/3/3 21:58
Author:WRZ
"""
from lxml import etree
def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
        Python dictionary holding XML contents.
    """
    # print(len(xml))
    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result0 = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        # xml中存在有多个object因此需要单独处理
        if child.tag != 'object':
            result0[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result0:  # 因为object可能有多个，所以需要放入列表里
                result0[child.tag] = []  # 第一个object建立空列表然后append，第二个object直接append
            result0[child.tag].append(child_result[child.tag])
    return {xml.tag: result0}


if __name__ == '__main__':

    xml_path = r'C:\Users\11733\Desktop\ArcgisPro\project_1\导出\labels\000000000.xml'
    with open(xml_path) as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = parse_xml_to_dict(xml)['annotation']
    print('object一共有{}个'.format(len(data['object'])))