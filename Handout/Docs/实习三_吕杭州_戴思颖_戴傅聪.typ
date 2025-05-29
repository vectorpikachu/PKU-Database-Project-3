#import "@preview/cuti:0.3.0": show-cn-fakebold
#show: show-cn-fakebold
#import "@preview/codly:1.3.0": *
#import "@preview/ansi-render:0.8.0": * // Render a terminal-like output.
#import "@preview/codly-languages:0.1.1": *

#show emph: set text(font: ("New Computer Modern", "KaiTi"), size: 12pt)

#let radius = 3pt
#let inset = 8pt

#let title = "实习三：非关系数据实习"
#let date = datetime.today()
#set text(
  lang: "zh",
  font: ("New Computer Modern", "SimSun"),
  region: "cn",
  size: 10pt,
)
#set page(
  "a4",
  margin: 1in,
	numbering: "第 1 页, 共 1 页",
	header: context {
		if counter(page).at(here()).first() > 1 [
		#grid(
			columns: (1fr, auto, 1fr),
			align: (left, center, right),
			[_北京大学_], [], [2025_年春季学期数据库概论_]
		)
		#v(-10pt)
		#line(length: 100%, stroke: 0.5pt)
	]}
)
#show raw.where(block: false): set text(font: "Maple Mono NF", size: 9pt, weight: "light")
#show raw.where(block: true): set text(font: "Maple Mono NF", size: 8pt, weight: "light")
#set heading(
  numbering: "1.1.1.1."
)
#set enum(
  numbering: "1..a..i."
)
#show link: this => {
	set text(bottom-edge: "bounds", top-edge: "bounds")
	text(this, fill: rgb("#433af4"))
}

#show: codly-init.with(
	
)

#codly(languages: codly-languages)


#align(center,
[#block(text(size: 16pt, [*#title*]))
#block(text(size: 12pt, [_吕杭州_ 2200013126 #h(1cm) _戴思颖_ 2200094811 #h(1cm) _戴傅聪_ 2100013061]))
#block(text(size: 12pt, [
	#date.year()_年_#date.month()_月_#date.day()_日_
]))])

#set par(
	justify: true,
	linebreaks: "optimized",
	first-line-indent: (amount: 2em, all: true)
)

#let output-block(body) = context {
  v(0pt, weak: true)
  ansi-render(
    body,
    radius: radius,
    inset: inset,
    width: 100%,
		font: none,
		size: 8pt,
		theme: terminal-themes.one-half-light,
  )
}


本次实习的目标是请同学们掌握一些典型的非关系数据类型在关系数据库中的处理手段。主要包括如下几方面的练习：
1. 递归查询
2. JSON和关系表的导入导出和基本查询
3. 向量数据库的使用

= 递归查询

下面是八王之乱的亲属关系示意图，我们先把它存入一个family表中，记录图中直接的父子关系。

#figure(
	align(center, image("./image.png", width: 80%)),
	caption: [八王之乱的亲属关系示意图]
)

```python
import sqlite3
import os

conn = sqlite3.connect('family.db')
cursor = conn.cursor()

cursor.execute('DROP TABLE IF EXISTS family;')
# Drop the family table if it already exists
cursor.execute('''
    CREATE TABLE family (
    father TEXT,
    son TEXT
);
''')
```
#output-block("<sqlite3.Cursor at 0x1da01bf9940>")

```python
# Insert the values 
insert_data = [
    ('司马防', '司马懿'),
    ('司马防', '司马孚'),
    ('司马防', '司马馗'),
    ('司马懿', '司马师'),
    ('司马懿', '司马昭'),
    ('司马懿', '司马亮'),
    ('司马懿', '司马伦'),
    ('司马孚', '司马瑰'),
    ('司马馗', '司马泰'),
    ('司马师', '司马攸'),
    ('司马昭', '司马炎'),
    ('司马瑰', '司马颙'),
    ('司马攸', '司马囧'),
    ('司马炎', '司马衷'),
    ('司马炎', '司马玮'),
    ('司马炎', '司马乂'),
    ('司马炎', '司马颖'),
    ('司马炎', '司马炽')
]

cursor.executemany('INSERT INTO family VALUES (?, ?)', insert_data)
conn.commit()
```

我们定义如下规则：

brother(X,Y):-father(Z,X),father(Z,Y). 有共同父亲的是兄弟。

ancestor(X,Y):-father(X,Y). 父亲是祖先。

ancestor(X,Y):-father(X,Z),ancestor(Z,Y).父亲的祖先是祖先。

请同学们完成如下SQL查询：

1) 找出所有的兄弟关系。

2) 使用递归查询找出所有祖先关系。

```python
#1. Brother rule: brother(X,Y) :- father(Z,X), father(Z,Y)
def find_brothers():
    cursor=conn.cursor()
    cursor.execute('''
    SELECT f1.son AS son1,f2.son AS son2
    FROM family f1
    JOIN family f2 on f1.father=f2.father
    WHERE son1 < son2
    ORDER BY son1,son2
        
    ''')
    brothers = cursor.fetchall()
    print("Brothers(无重复):")
    for b in brothers:
        print(f"({b[0]},{b[1]})")
```

```python
# 2. Ancestor rules:
#    ancestor(X,Y) :- father(X,Y)
#    ancestor(X,Y) :- father(X,Z), ancestor(Z,Y)
def find_ancestors():
    # This requires a recursive query
    cursor = conn.cursor()
    cursor.execute('''
    WITH RECURSIVE ancestor(ancestor, descendant) AS (
        -- Base case: direct father-son relationships
        SELECT father, son FROM family
        
        UNION
        
        -- Recursive case: father of someone who is already an ancestor
        SELECT f.father, a.descendant
        FROM family f
        JOIN ancestor a ON f.son = a.ancestor
    )
    SELECT * FROM ancestor ORDER BY ancestor, descendant
    ''')
    ancestors = cursor.fetchall()
    print("\nAncestors:")
    print("左边是右边的祖先：")
    for a in ancestors:
        print(f"({a[0]},{a[1]})")
```

```python
# Execute the queries
find_brothers()
find_ancestors()
```

#output-block("Brothers(无重复):
(司马乂,司马炽)
(司马乂,司马玮)
(司马乂,司马衷)
(司马乂,司马颖)
(司马亮,司马伦)
(司马亮,司马师)
(司马亮,司马昭)
(司马伦,司马师)
(司马伦,司马昭)
(司马孚,司马懿)
(司马孚,司马馗)
(司马师,司马昭)
(司马懿,司马馗)
(司马炽,司马玮)
(司马炽,司马衷)
(司马炽,司马颖)
(司马玮,司马衷)
(司马玮,司马颖)
(司马衷,司马颖)

Ancestors:
左边是右边的祖先：
(司马孚,司马瑰)
(司马孚,司马颙)
(司马师,司马囧)
(司马师,司马攸)
(司马懿,司马乂)
(司马懿,司马亮)
(司马懿,司马伦)
(司马懿,司马囧)
(司马懿,司马师)
(司马懿,司马攸)
(司马懿,司马昭)
(司马懿,司马炎)
(司马懿,司马炽)
(司马懿,司马玮)
(司马懿,司马衷)
(司马懿,司马颖)
(司马攸,司马囧)
(司马昭,司马乂)
(司马昭,司马炎)
(司马昭,司马炽)
(司马昭,司马玮)
(司马昭,司马衷)
(司马昭,司马颖)
(司马炎,司马乂)
(司马炎,司马炽)
(司马炎,司马玮)
(司马炎,司马衷)
(司马炎,司马颖)
(司马瑰,司马颙)
(司马防,司马乂)
(司马防,司马亮)
(司马防,司马伦)
(司马防,司马囧)
(司马防,司马孚)
(司马防,司马师)
(司马防,司马懿)
(司马防,司马攸)
(司马防,司马昭)
(司马防,司马泰)
(司马防,司马炎)
(司马防,司马炽)
(司马防,司马玮)
(司马防,司马瑰)
(司马防,司马衷)
(司马防,司马颖)
(司马防,司马颙)
(司马防,司马馗)
(司马馗,司马泰)")

```python
#GROUP_CONCAT
def find_ancestors():
    cursor = conn.cursor()
    cursor.execute('''
    WITH RECURSIVE ancestor(ancestor, descendant) AS (
        -- 基础情况：直接的父子关系
        SELECT father, son FROM family
        
        UNION
        
        -- 递归情况：祖先的祖先也是祖先
        SELECT f.father, a.descendant
        FROM family f
        JOIN ancestor a ON f.son = a.ancestor
    )
    SELECT 
        ancestor,
        GROUP_CONCAT(descendant, '、') AS descendants
    FROM ancestor
    GROUP BY ancestor
    ORDER BY ancestor
    ''')
    
    ancestors = cursor.fetchall()
    print("\n祖先及其后辈列表：")
    print("===========================================================================")
    for ancestor, descendants in ancestors:
        print(f"{ancestor}：{descendants}")

# 调用函数
find_ancestors()
```

#output-block("祖先及其后辈列表：
==================================================================================
司马孚：司马瑰、司马颙
司马师：司马攸、司马囧
司马懿：司马师、司马昭、司马亮、司马伦、司马攸、司马炎、司马囧、司马衷、司马玮、司马乂、司马颖、司马炽
司马攸：司马囧
司马昭：司马炎、司马衷、司马玮、司马乂、司马颖、司马炽
司马炎：司马衷、司马玮、司马乂、司马颖、司马炽
司马瑰：司马颙
司马防：司马懿、司马孚、司马馗、司马师、司马昭、司马亮、司马伦、司马瑰、司马泰、司马攸、司马炎、司马颙、司马囧、司马衷、司马玮、司马乂、司马颖、司马炽
司马馗：司马泰")

```python
conn.close()
```


= JSON操作

同学们选择我们课程的某个章节，结合大模型生成课程某个章节的内容，并以JSON格式输出，然后将其存入一个表中，定义一个JSON类型的字段来存储大模型的输出内容。

同学们的任务包括如下两项：

1. 自己设计两个查询实例，分别查询特定字段的值以及特定数组的元素数目。
2. 把JSON的嵌套结构展开，存入一个1NF平面表中。

```python
import sqlite3
import json

# 创建数据库连接
conn = sqlite3.connect('database_course.db')
c = conn.cursor()

# 创建存储JSON的原始表
c.execute('''CREATE TABLE IF NOT EXISTS chapter_json (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content JSON
             )''')
```

#output-block("<sqlite3.Cursor at 0x175a50b0540>")

```python
# 示例JSON数据
data = {
    "chapter": "关系规范化理论",
    "sections": [
        {
            "section_title": "基本概念与问题引入",
            "knowledge_points": [
                {
                    "key_point": "关系模式设计问题",
                    "explanation": "非规范化关系模式会导致数据冗余、插入异常、更新异常和删除异常。例如学生选课表中存储系主任信息时，会导致系主任信息重复存储（冗余）且无法独立维护（异常）",
                    "qa": {
                        "question": "非规范化关系模式可能引发哪些问题？请举例说明。",
                        "answer": "数据冗余（如系主任信息重复存储）、插入异常（无法单独插入未选课学生的系信息）、更新异常（修改系主任需更新多条记录）、删除异常（删除最后一条学生记录会丢失系信息）"
                    }
                },
                {
                    "key_point": "函数依赖",
                    "explanation": "描述属性间逻辑关系的约束，包括完全依赖、部分依赖和传递依赖。如学号→系名是完全依赖，(学号,课程)→成绩是部分依赖，学号→系主任是传递依赖",
                    "qa": {
                        "question": "什么是传递函数依赖？请用学生-系-系主任的例子说明。",
                        "answer": "若存在学号→系名，系名→系主任，且系名↛学号，则系主任传递依赖于学号"
                    }
                }
            ]
        },
        {
            "section_title": "范式体系",
            "knowledge_points": [
                {
                    "key_point": "第一范式（1NF）",
                    "explanation": "属性值不可再分，消除重复组。如将包含多值的地址字段拆分为省、市、街道",
                    "qa": {
                        "question": "判断表结构是否满足1NF：商品表(商品ID, 商品名称, 规格['红色','L'])",
                        "answer": "不满足1NF，'规格'字段包含多个值，需拆分为独立属性或新建规格表"
                    }
                },
                # 其他知识点...
            ]
        },
        # 其他章节...
    ],
    "question_bank": [
        {
            "question": "Armstrong公理包含哪些基本规则？",
            "answer": "自反律（若Y⊆X则X→Y）、增广律（X→Y则XZ→YZ）、传递律（X→Y且Y→Z则X→Z）"
        },
        # 其他问题...
    ]
}
```

```python
# 插入JSON数据
c.execute("INSERT INTO chapter_json (content) VALUES (?)", 
          (json.dumps(data, ensure_ascii=False),))
conn.commit()
```

```python
# 查询1：获取特定字段值（所有章节标题）
print("查询1：所有章节标题")
c.execute('''SELECT json_extract(content, '$.chapter') 
             FROM chapter_json''')
print(c.fetchone()[0])
```

#output-block("查询1：所有章节标题
关系规范化理论")

```python
# 查询2：统计特定数组元素数目（每个section的知识点数量）
print("\n查询2：各章节知识点数量")
c.execute('''SELECT json_extract(s.value, '$.section_title'),
                    json_array_length(s.value, '$.knowledge_points')
             FROM chapter_json, 
                  json_each(chapter_json.content, '$.sections') AS s''')
for row in c.fetchall():
    print(f"{row[0]}: {row[1]}个知识点")
```

#output-block("查询2：各章节知识点数量
基本概念与问题引入: 2个知识点
范式体系: 1个知识点")

```python
# 创建规范化平面表（1NF）
c.execute('''CREATE TABLE IF NOT EXISTS normalized_chapter (
                chapter TEXT,
                section_title TEXT,
                key_point TEXT,
                explanation TEXT,
                question TEXT,
                answer TEXT
             )''')

# 展开JSON结构插入平面表
c.execute('''INSERT INTO normalized_chapter
             SELECT json_extract(content, '$.chapter'),
                    json_extract(s.value, '$.section_title'),
                    json_extract(kp.value, '$.key_point'),
                    json_extract(kp.value, '$.explanation'),
                    json_extract(kp.value, '$.qa.question'),
                    json_extract(kp.value, '$.qa.answer')
             FROM chapter_json,
                  json_each(chapter_json.content, '$.sections') AS s,
                  json_each(s.value, '$.knowledge_points') AS kp''')
conn.commit()
```

```python
# 验证规范化表
print("\n规范化表数据示例:")
c.execute("SELECT * FROM normalized_chapter LIMIT 2")
for row in c.fetchall():
    print(row)

# 创建问题库表
c.execute('''CREATE TABLE IF NOT EXISTS question_bank (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter TEXT,
                question TEXT,
                answer TEXT
             )''')

# 插入问题库数据
c.execute('''INSERT INTO question_bank (chapter, question, answer)
             SELECT json_extract(content, '$.chapter'),
                    json_extract(q.value, '$.question'),
                    json_extract(q.value, '$.answer')
             FROM chapter_json,
                  json_each(chapter_json.content, '$.question_bank') AS q''')
conn.commit()
```

#output-block("规范化表数据示例:
('关系规范化理论', '基本概念与问题引入', '关系模式设计问题', '非规范化关系模式会导致数据冗余、插入异常、更新异常和删除异常。例如学生选课表中存储系主任信息时，会导致系主任信息重复存储（冗余）且无法独立维护（异常）', '非规范化关系模式可能引发哪些问题？请举例说明。', '数据冗余（如系主任信息重复存储）、插入异常（无法单独插入未选课学生的系信息）、更新异常（修改系主任需更新多条记录）、删除异常（删除最后一条学生记录会丢失系信息）')
('关系规范化理论', '基本概念与问题引入', '函数依赖', '描述属性间逻辑关系的约束，包括完全依赖、部分依赖和传递依赖。如学号→系名是完全依赖，(学号,课程)→成绩是部分依赖，学号→系主任是传递依赖', '什么是传递函数依赖？请用学生-系-系主任的例子说明。', '若存在学号→系名，系名→系主任，且系名↛学号，则系主任传递依赖于学号')")

= 向量数据库实习设计

这部分我们的实习设计的比较简单，就是熟悉一下向量嵌入，以及数据库里面向量相似性检索的插件就可以了。
我们提供了一个Jupyter文件，里面包括SQLite里面向量插件的使用，以及生成向量嵌入的Python包的使用。

实验用的数据集放在sts-b-train.txt中，格式如下，前面是两个句子，后面的数字代表这两个句子的相似性。因为这是一个训练集，所以相似值已经给定了，同学们要做的是生成两个句子的向量嵌入，然后调用SQLite的向量插件来计算它们的相似性，并和现有的相似值做个比较。可以截取一部分数据，另外也可以用其他数据库的向量插件。

```text
一架飞机要起飞了。  一架飞机正在起飞。  5
一个男人在吹一支大笛子。    一个人在吹长笛。    3
三个人在下棋。  两个人在下棋。  2
一个人在拉大提琴。  一个坐着的人正在拉大提琴。  4
有些人在战斗。  两个人在打架。  4
一个男人在抽烟。    一个男人在滑冰。    0
那人在弹钢琴。  那人在弹吉他。  1
一个男人在弹吉他和唱歌。    一位女士正在弹着一把原声吉他，唱着歌。  2
一个人正把一只猫扔到天花板上。  一个人把一只猫扔在天花板上。    5
```

数据库sentence.db schema如下：

sentence(sid, sentence1, sentence2, similar_score, sen1_vector, sen2_vector, vecSim_score)

分别是句子id，句子1，句子2，句子相似度得分，句子1嵌入向量，句子2嵌入向量，向量相似度得分）

任务要求：

调用嵌入模型：可以使用sqlite-vss或sqlite-vec或调用本地中文嵌入模型（huggingface等拉取），对每个数据库中每个元组的两个句子分别进行向量嵌入，并计算向量相似度得分。更新数据库sentence.db的后三列。

输出要求：

按照这样的方法：

```python
df = pd.read_sql_query("""
    SELECT sid, sentence1, sentence2, similar_score, sen1_vector, sen2_vector, vecSim_score FROM sentence
""", conn)
print(df.head(20))
```
输出数据库的前20个元组。但请注意，助教评分时可能会运行你的代码，抽查其他元组结果。

```python
import sqlite3
import os
import pandas as pd
```

```python
# 文件路径修改成自己的文件路径
DATA_FILE = 'sts-b-train.txt'
DB_FILE = 'sentence.db'

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
```

```python
cursor.execute('''
CREATE TABLE IF NOT EXISTS sentence (
    sid INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence1 TEXT NOT NULL,
    sentence2 TEXT NOT NULL,
    similar_score REAL NOT NULL,
    sen1_vector BLOB,
    sen2_vector BLOB,
    vecSim_score REAL
)
''')
conn.commit()
```

```python
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        sentence1, sentence2, score = parts
        cursor.execute('''
            INSERT INTO sentence (sentence1, sentence2, similar_score)
            VALUES (?, ?, ?)
        ''', (sentence1, sentence2, float(score)))
conn.commit()
```

```python
df = pd.read_sql_query("""
    SELECT sid, sentence1, sentence2, similar_score, sen1_vector, sen2_vector, vecSim_score FROM sentence
""", conn)
print(df.head(10))
```

#no-codly()[
	```
	   sid          sentence1                  sentence2  similar_score  \
0    1          一架飞机要起飞了。                  一架飞机正在起飞。            5.0   
1    2       一个男人在吹一支大笛子。                   一个人在吹长笛。            3.0   
2    3  一个人正把切碎的奶酪撒在比萨饼上。  一个男人正在把切碎的奶酪撒在一块未煮好的比萨饼上。            3.0   
3    4            三个人在下棋。                    两个人在下棋。            2.0   
4    5          一个人在拉大提琴。              一个坐着的人正在拉大提琴。            4.0   
5    6            有些人在战斗。                    两个人在打架。            4.0   
6    7           一个男人在抽烟。                   一个男人在滑冰。            0.0   
7    8            那人在弹钢琴。                    那人在弹吉他。            1.0   
8    9       一个男人在弹吉他和唱歌。        一位女士正在弹着一把原声吉他，唱着歌。            2.0   
9   10    一个人正把一只猫扔到天花板上。       一个人把一只猫扔在天花板上。            5.0   

  sen1_vector sen2_vector vecSim_score  
0        None        None         None  
1        None        None         None  
2        None        None         None  
3        None        None         None  
4        None        None         None  
5        None        None         None  
6        None        None         None  
7        None        None         None  
8        None        None         None  
9        None        None         None  
	```
]

```python
from sentence_transformers import SentenceTransformer

# 加载模型, 需要先下载到本地文件夹下
model = SentenceTransformer('./text2vec-base-chinese')
```

```python
cursor = conn.cursor()

cursor.execute("SELECT sid, sentence1, sentence2 FROM sentence")
rows = cursor.fetchall()

# 提取句子和对应的 sid
sid_list = []
sentences1 = []
sentences2 = []

for row in rows:
    sid, s1, s2 = row
    sid_list.append(sid)
    sentences1.append(s1)
    sentences2.append(s2)

# 生成嵌入向量
embeddings1 = model.encode(sentences1, batch_size=64, show_progress_bar=True)
embeddings2 = model.encode(sentences2, batch_size=64, show_progress_bar=True)
```

#output-block("Batches: 100%|██████████| 82/82 [00:44<00:00,  1.85it/s]
Batches: 100%|██████████| 82/82 [00:44<00:00,  1.85it/s]")

```python
import numpy as np
def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
```

```python
for sid, vec1, vec2 in zip(sid_list, embeddings1, embeddings2):
    # 计算余弦相似度
    sim_score = cosine_similarity(vec1, vec2)

    # 将向量转换为二进制格式
    vec1_blob = vec1.tobytes()
    vec2_blob = vec2.tobytes()

    # 更新数据库记录
    cursor.execute("""
        UPDATE sentence
        SET sen1_vector = ?, sen2_vector = ?, vecSim_score = ?
        WHERE sid = ?
    """, (vec1_blob, vec2_blob, sim_score, sid))

# 提交更改
conn.commit()
```

```python
df = pd.read_sql_query("""
    SELECT sid, sentence1, sentence2, similar_score, sen1_vector, sen2_vector, vecSim_score FROM sentence
""", conn)
print(df.head(20))
```

#no-codly(
	```
	    sid           sentence1                  sentence2  similar_score  \
0     1           一架飞机要起飞了。                  一架飞机正在起飞。            5.0   
1     2        一个男人在吹一支大笛子。                   一个人在吹长笛。            3.0   
2     3   一个人正把切碎的奶酪撒在比萨饼上。  一个男人正在把切碎的奶酪撒在一块未煮好的比萨饼上。            3.0   
3     4             三个人在下棋。                    两个人在下棋。            2.0   
4     5           一个人在拉大提琴。              一个坐着的人正在拉大提琴。            4.0   
5     6             有些人在战斗。                    两个人在打架。            4.0   
6     7            一个男人在抽烟。                   一个男人在滑冰。            0.0   
7     8             那人在弹钢琴。                    那人在弹吉他。            1.0   
8     9        一个男人在弹吉他和唱歌。        一位女士正在弹着一把原声吉他，唱着歌。            2.0   
9    10     一个人正把一只猫扔到天花板上。             一个人把一只猫扔在天花板上。            5.0   
10   11        那人用棍子打了另一个人。            那人用棍子打了另一个人一巴掌。            4.0   
11   12            一个人在吹长笛。                  一个男人在吹竹笛。            3.0   
12   13           一个人在叠一张纸。                   有人在叠一张纸。            4.0   
13   14  一条狗正试图把培根从他的背上弄下来。             一只狗正试图吃它背上的培根。            3.0   
14   15          北极熊在雪地上滑行。               一只北极熊在雪地上滑行。            5.0   
15   16            一个女人在写作。                   一个女人在游泳。            0.0   
16   17        一只猫在婴儿的脸上摩擦。                   一只猫在蹭婴儿。            3.0   
17   18              那人在骑马。                    一个人骑着马。            5.0   
18   19           一个人往锅里倒油。                 一个人把酒倒进锅里。            3.0   
19   20           一个男人在弹吉他。                  一个女孩在弹吉他。            2.0   

                                          sen1_vector  \
0   b'tw\xbb\xbe\xbc=#>\xfc\x86\xac=lG->6\x8aw>wzU...   
1   b'\t\x03\xa5\xbc\xf0\xdd\x92\xbe\xc1&\x86\xbf\...   
2   b'\xac\xc1\n?\x0b\xbb\x0b?\x8dz\x1b\xbe\x83\xd...   
3   b'\xd5C\xfd>T\xfa#\xbf\xac\xcd\x04?\x18\x1e\x9...   
4   b'Q\x92\xde;;q\x04?\xb0\x1d\x83\xbe\xd3\x08\x9...   
5   b'\x8c\xcf\xd5\xbe\xc0M`\xbe\xe4d\xa9\xbeu\x19...   
6   b'W|I\xbe`\xfaz?Z\x1dq?zP\xab?\xb0@\x12\xbdH\x...   
7   b's\xb8V?\xcbj\xf0=\x97\xeeh\xbe\x84|5?$h\xe7=...   
8   b'\xce\x80\xe4>%\xaf\xf6>\x83\xbe7?\xc8H"?h\xa...   
9   b'U1\xc5\xbd}p\x98=\x17\xdc\xef=jWR\xbe\xc7\xc...   
10  b'&\x88\xc4={\x19\t\xbe\x19\xb9\x99\xbeI\xcd ?...   
11  b'\x92c\x90\xbe\xd9\xb8\xb1\xbe\xdc/\x00\xbf>`...   
12  b'\x17B\xb9\xbe\xae:\x06>\x9c\x9c\x03?\xd6\xd5...   
13  b'\xc8O\xc6>\x0e\xeaS\xbf\xb7=D>2\xba\xaa\xbev...   
14  b'K\xb2P\xbd+\xbb\x08\xbf};\x88\xbe\xfc\xab\xc...   
15  b'Q\xb1&?\xee\x0b\xef\xbd@Q(?\x85wb?\xc2U\xab\...   
16  b'\x03M\xb4\xbd7\x9c\x9f\xbdn\xe0\x14>>\x81k\x...   
17  b'\x13\xa3\xe0<\xb8\x0bX\xbfUG\x01?\xff\x15\x9...   
18  b'\xb2\x8a\x19?]Q\xae?\xf9\x16-\xbd\xc7\x82^>$...   
19  b'\x97\x85-?\xe2w&?\'\xa6\xfc>\x1c\xf62>\xbd\x...   

                                          sen2_vector  vecSim_score  
0   b'_\x96\x1e\xbf\x11_y>\xec\x88\x03?\xdc(\xf1=\...      0.956823  
1   b'\x92c\x90\xbe\xd9\xb8\xb1\xbe\xdc/\x00\xbf>`...      0.884109  
2   b'\x16\x0c\x0b?\xc5\xbd\x0e?\x9aU:\xbe\xf4\xb8...      0.974107  
3   b'\xf2&\xe4>\'{;\xbf\xd5]J?\xb7)S?\x079X?h\x84...      0.747162  
4   b'P\xfe\x8f==\x11\x06?)\xa0\xb1\xbe\xf9\xb5\x8...      0.961783  
5   b'\x12h\xe9\xbc\x93o\x9c\xbe)&_\xbe"\xd8\x17?\...      0.846043  
6   b'\xba\xdb\xc1?\xa6P\xa9=\x9a\xcd\xc4\xbd\xf0m...      0.231173  
7   b'IP\x07?B\xbb\xc5>\xc0;\xef>\xf5\x93\x87>i\xf...      0.484320  
8   b'\xdbi>;\xdb\xa3<\xbb\xcb6\x85\xbd{\xdf\x8b?\...      0.734604  
9   b'\xb8\xb8\x14\xbd\xca\x81\xc1\xbd2\x86\xc9=\x...      0.983240  
10  b'@\xcd\x88>\xa2\xc2\x8c>\xd0\x18,\xbe\x9e\xcc...      0.909791  
11  b']\xd0\xb6>\x89\x89\x1a=\xc3lq\xbf\x81\xf6\xc...      0.836796  
12  b'\x1f\xd2\xaa\xbe\xf8\xa3j> \xb9\xfa>\x88\x9b...      0.979772  
13  b'r\x93\xa8>\xfa\x1e\x95\xbew\xd9\xfa\xbe\x89\...      0.798483  
14  b'\xae\xa9\xb7\xbc\x10\xb2\xda\xbe\x0e\x9d\x83...      0.985583  
15  b'H\xf8\x95?Nm\xd8>\xa2L\x0e\xbf\xc3\x95\x8e>@...      0.425817  
16  b'A\xac\xa7\xbea\xba=\xbf*A\x80\xbf\x9b=\xfd\x...      0.798582  
17  b'=t\t=DIS\xbf\xb7E*>\x9b\xabw?\x8d\x1e)\xbe\x...      0.944069  
18  b'8=E?D\x82\xdc?\xd5$\r\xbc\xad\x1f<>\xc9q\xe5...      0.687595  
19  b'd\xaeA?\xf6X\xeb<PL\x94>\xcd\xa3\x9e>\xe9\xc...      0.706035  
	```
)

```python
conn.close()
```