## 推荐模块工作

#### 1.问题定义
	用户输入：
	* N: int 游玩总天数。
	     e.g. N = 10
	     
	* C: list 用户想去的城市。 
		  e.g. C = {大阪}
	
	算法输出：
	* O: list 城市推荐结果(包括中间统计结果与历史数据)
		  e.g. O = [
		  				'大阪': 29.3, '京都': 1.9, '箱根': 9.8, 
		  				'小田原': 0.7, '奈良': 8.0, '三鹰市': 6.7}
		  			 ]
		  			 

<img src="https://raw.githubusercontent.com/pqy000/debug1/master/Figure3.png" width = "550" alt="操作流程" align=center />

#### 2.解决问题
- 将Kernal Density Estimation的bandwidth带宽调低，防止时间波动过大，消除不确定性
- 剔除机场城市，将游玩天数对应的城市的搜索空间缩小
- 查找用户输入线路的历史记录
- 输出生成城市数目时的中间过程

<img src="https://raw.githubusercontent.com/pqy000/debug1/master/Figure4.png" width = "550" alt="操作流程" align=center />

代码已上传到github上..

#### 3.存在问题

在数据预处理阶段，对机场数据进行剔除，并将其分为5类json格式的文件：

(path1-3.json, path4-5.json, path6-7.json, path8-10.json, pathother.json)

现存问题:

- 给定的机场文件airport_jp.json中有些不合理处，有将游玩城市加进去的情况。剔除数据后会导致游玩城市未出现在线路

<img src="https://raw.githubusercontent.com/pqy000/debug1/master/Figure1.png" width = "350" alt="操作流程" align=center />
 
- 火车站 汽车站等其他交通城市 需要提供数据文件处理，因为游玩时间过短，可能需要剔除，不占总城市数

<img src="https://raw.githubusercontent.com/pqy000/debug1/master/Figure2.png" width = "350" alt="操作流程" align=center />

- 推荐城市的权重现在是根据在线路中出现的次数转化为概率，按照概率大小选取..暂定评价城市的知名度还需要有更好的量化指标




