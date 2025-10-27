import re
import codecs
from typing import List
import math

'''
针对代码更改的几种不同分词方式
'''

# 特殊的关键词，函数名或者变量名，不对其进行抽象化
keywords = frozenset({'__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
                      '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
                      '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
                      '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
                      '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
                      '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
                      'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
                      'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
                      'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
                      'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
                      'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                      'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
                      'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
                      'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
                      'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL', 'STRING', 'CHARACTER', 'include'})
keywords_1 = frozenset({'memcpy', 'wmemcpy', '_memccpy', 'memmove', 'wmemmove', 'memset', 'wmemset', 'memcmp', 'wmemcmp', 'memchr',
                        'wmemchr', 'strncpy', 'lstrcpyn', 'wcsncpy', 'strncat', 'bcopy', 'cin', 'strcpy', 'lstrcpy', 'wcscpy', '_tcscpy',
                        '_mbscpy', 'CopyMemory', 'strcat', 'lstrcat', 'fgets', 'main', '_main', '_tmain', 'Winmain', 'AfxWinMain', 'getchar',
                        'getc', 'getch', 'getche', 'kbhit', 'stdin', 'm_lpCmdLine', 'getdlgtext', 'getpass', 'istream.get', 'istream.getline',
                        'istream.peek', 'istream.putback', 'streambuf.sbumpc', 'streambuf.sgetc', 'streambuf.sgetn', 'streambuf.snextc',
                        'streambuf.sputbackc',
                        'SendMessage', 'SendMessageCallback', 'SendNotifyMessage', 'PostMessage', 'PostThreadMessage', 'recv', 'recvfrom',
                        'Receive',
                        'ReceiveFrom', 'ReceiveFromEx', 'CEdit.GetLine', 'CHtmlEditCtrl.GetDHtmlDocument', 'CListBox.GetText',
                        'CListCtrl.GetItemText',
                        'CRichEditCtrl.GetLine', 'GetDlgItemText', 'CCheckListBox.GetCheck', 'DISP_FUNCTION', 'DISP_PROPERTY_EX', 'getenv',
                        'getenv_s', '_wgetenv',
                        '_wgetenv_s', 'snprintf', 'vsnprintf', 'scanf', 'sscanf', 'catgets', 'gets', 'fscanf', 'vscanf', 'vfscanf', 'printf',
                        'vprintf', 'CString.Format',
                        'CString.FormatV', 'CString.FormatMessage', 'CStringT.Format', 'CStringT.FormatV', 'CStringT.FormatMessage',
                        'CStringT.FormatMessageV',
                        'vsprintf', 'asprintf', 'vasprintf', 'fprintf', 'sprintf', 'syslog', 'swscanf', 'sscanf_s', 'swscanf_s', 'swprintf',
                        'malloc',
                        'readlink', 'lstrlen', 'strchr', 'strcmp', 'strcoll', 'strcspn', 'strerror', 'strlen', 'strpbrk', 'strrchr', 'strspn',
                        'strstr',
                        'strtok', 'strxfrm', 'kfree', '_alloca', '_strncpy*', '_tcsncpy*', '_mbsnbcpy*', '_wcsncpy*', '_strncat*', '_mbsncat*', 'wcsncat*', 'CEdit.Get*',
                        'CRichEditCtrl.Get*',
                        'CComboBox.Get*', 'GetWindowText*', 'istream.read*', 'Socket.Receive*', 'DDX_*', '_snprintf*',
                        '_snwprintf*', '*malloc'})


# holds known non-user-defined functions; immutable set
main_set = frozenset({'main'})
# arguments in main function; immutable set
main_args = frozenset({'argc', 'argv'})

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '**',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '&',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':',
    '{', '}', '!', '~',
}

def to_regex(lst):
    return r'|'.join([f"({re.escape(el)})" for el in lst])

regex_split_operators = to_regex(operators3) + to_regex(operators2) + to_regex(operators1)


def get_sub_word(code):
    subword = re.sub('[^a-zA-Z0-9]', ' ', code)
    subword = re.sub(' +', ' ', subword)
    subword = re.sub(r"([A-Z][a-z]+)", r" \1", subword)
    subword = re.sub(r"([A-Z]{2,})", r" \1", subword)
    return subword

def remove_comment(code):
    in_comment = 0
    output = []
    for line in code.splitlines():
        if in_comment == 0:
            if line.find("/*") != -1:
                if line.find("*/") == -1:
                    in_comment = 1
            else:
                if line.find("//") != -1:
                    if line.find("//") > 0:
                        line = line[:line.find("//")]
                        output.append(line)
                else:
                    output.append(line)
        else:
            if line.find("*/") != -1:
                in_comment = 0
    return " ".join(output)

def tokenizer_subword(code):
    # 将函数名与变量名分割为subword
    fun_symbols = {}
    var_symbols = {}
    cg = []
    if isinstance(code, str):
        rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
        rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()')

        code = re.sub(r'[^\x00-\x7f]', r'', code)
        code = remove_comment(code)
        code = re.sub(r'"(.|\n)*?"', '""', code)
        code = re.sub(r"'.*?'", "''", code)

        user_funs = rx_fun.findall(code)
        user_vars = rx_var.findall(code)

        for user_fun in user_funs:
            if user_fun not in fun_symbols and user_fun not in main_set and user_fun not in keywords and user_fun not in keywords_1:
                subword= re.sub('[^a-zA-Z0-9]', ' ', user_fun)
                subword = re.sub(' +', ' ', subword)
                subword = re.sub(r"([A-Z][a-z]+)", r" \1", subword)
                fun_symbols[user_fun] = subword
                code = re.sub(r'\b(' + user_fun + r')\b(?=\s*\()', fun_symbols[user_fun], code)
        for user_var in user_vars:
            if user_var not in var_symbols and user_var not in main_args and user_var not in keywords and user_var not in keywords_1:
                subword = re.sub('[^a-zA-Z0-9]', ' ', user_var)
                subword = re.sub(' +', ' ', subword)
                subword = re.sub(r"([A-Z][a-z]+)", r" \1", subword)
                var_symbols[user_var] = subword
                code = re.sub(r'\b(' + user_var + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', \
                                        var_symbols[user_var], code)
            # 去除换行和制表符
        code = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(\\r)', ' ', code)
        # 分词
        splitter = r' +|' + regex_split_operators + r'|(\/)|(\;)|(\-)|(\*)'
        cg = re.split(splitter, code)
        # 删除None对象
        cg = list(filter(None, cg))
        cg = list(filter(str.strip, cg))
    return cg

def tokenizer_(code):
    # 去除换行和制表符
    code = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(\\r)', ' ', code)
    code = re.sub(r'[^\x00-\x7f]', r'', code)
    code = remove_comment(code)
    code = re.sub(r'"(.|\n)*?"', '""', code)
    code = re.sub(r"'.*?'", "''", code)
    # 分词
    splitter = r' +|' + regex_split_operators + r'|(\/)|(\;)|(\-)|(\*)'
    cg = re.split(splitter, code)
    # 删除None对象
    cg = list(filter(None, cg))
    cg = list(filter(str.strip, cg))
    return cg
def abstract_code(add_code, del_code):
    '''抽象化，包括删除注释，替换字符串，替换函数名和变量名'''
    abstract_code = []
    # 删除注释
    add_code = re.sub(r"/\*(\s|\S)*?\*/|/\*.*|^\s*\*.*|//.*", "", add_code, flags=re.MULTILINE)
    del_code = re.sub(r"/\*(\s|\S)*?\*/|/\*.*|^\s*\*.*|//.*", "", del_code, flags=re.MULTILINE)
    # 替换所有字符串为STRING, 替换所有字符为CHARACTER
    add_code = re.sub(r'"(.|\n)*?"', 'STRING', add_code)
    del_code = re.sub(r'"(.|\n)*?"', 'STRING', del_code)
    add_code = re.sub(r"'.*?'", "CHARACTER", add_code)
    del_code = re.sub(r"'.*?'", "CHARACTER", del_code)
    abstract_code.append(add_code)
    abstract_code.append(del_code)
    abstract_code = clean_gadget(abstract_code)
    return abstract_code[0], abstract_code[1]
# input is a list of string lines
def clean_gadget(gadget):
    # 使用VAR和FUN替换函数于变量名
    fun_symbols = {}
    # dictionary; map variable name to symbol name + number
    var_symbols = {}

    fun_count = 1
    var_count = 1

    # regular expression to catch multi-line comment
    rx_comment = re.compile('\*/\s*$')
    # regular expression to find function name candidates

    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    # regular expression to find variable name candidates
    #rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()')

    # final cleaned gadget output to return to interface
    cleaned_gadget = []

    for line in gadget:
        # process if not the header line and not a multi-line commented line
        # if rx_comment.search(line) is None:
        # remove all string literals (keep the quotes)
        nostrlit_line = re.sub(r'".*?"', '""', line)
        # remove all character literals
        nocharlit_line = re.sub(r"'.*?'", "''", nostrlit_line)

        # replace any non-ASCII characters with empty string
        ascii_line = re.sub(r'[^\x00-\x7f]', r'', nocharlit_line)


        # return, in order, all regex matches at string list; preserves order for semantics
        user_fun = rx_fun.findall(ascii_line)
        user_var = rx_var.findall(ascii_line)

        # Could easily make a "clean gadget" type class to prevent duplicate functionality
        # of creating/comparing symbol names for functions and variables in much the same way.
        # The comparison frozenset, symbol dictionaries, and counters would be class scope.
        # So would only need to pass a string list and a string literal for symbol names to
        # another function.
        for fun_name in user_fun:
            if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0 and len({fun_name}.difference(keywords_1)) != 0:
                # DEBUG
                #print('comparing ' + str(fun_name + ' to ' + str(main_set)))
                #print(fun_name + ' diff len from main is ' + str(len({fun_name}.difference(main_set))))
                #print('comparing ' + str(fun_name + ' to ' + str(keywords)))
                #print(fun_name + ' diff len from keywords is ' + str(len({fun_name}.difference(keywords))))
                ###
                # check to see if function name already in dictionary
                if fun_name not in fun_symbols.keys():
                    fun_symbols[fun_name] = 'FUN' + str(fun_count)
                    fun_count += 1
                # ensure that only function name gets replaced (no variable name with same
                # identifier); uses positive lookforward
                ascii_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], ascii_line)

        for var_name in user_var:
            # next line is the nuanced difference between fun_name and var_name
            if len({var_name}.difference(keywords)) != 0 and len({var_name}.difference(main_args)) != 0  and len({var_name}.difference(keywords_1)) != 0:
                # DEBUG
                #print('comparing ' + str(var_name + ' to ' + str(keywords)))
                #print(var_name + ' diff len from keywords is ' + str(len({var_name}.difference(keywords))))
                #print('comparing ' + str(var_name + ' to ' + str(main_args)))
                #print(var_name + ' diff len from main args is ' + str(len({var_name}.difference(main_args))))
                ###
                # check to see if variable name already in dictionary
                if var_name not in var_symbols.keys():
                    var_symbols[var_name] = 'VAR' + str(var_count)
                    var_count += 1
                # ensure that only variable name gets replaced (no function name with same
                # identifier); uses negative lookforward
                ascii_line = re.sub(r'\b(' + var_name + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', \
                                    var_symbols[var_name], ascii_line)

        cleaned_gadget.append(ascii_line)
    # return the list of cleaned lines
    # cleaned_gadget = [i for i in cleaned_gadget if i != '']
    return cleaned_gadget

def BPE(tokens = None, train = False, save_path=None):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    from tokenizers.trainers import BpeTrainer
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    from tokenizers import pre_tokenizers
    from tokenizers.pre_tokenizers import Whitespace, Split
    from tokenizers.normalizers import Lowercase
    from tokenizers import normalizers
    normalizer = normalizers.Sequence([Lowercase()])

    pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Split("bad", "removed"),Split(regex_split_operators, "isolated")])
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    if train:
        tokenizer.train_from_iterator(tokens.tokens, trainer)
        tokenizer.save(save_path)
        return tokenizer
    tokenizer = Tokenizer.from_file(save_path)
    return tokenizer

if __name__ == '__main__':

    code = """ printf("%d\n", x);"""
    print(tokenizer_subword(code))




