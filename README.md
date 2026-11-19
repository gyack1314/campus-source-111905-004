# 实验阵列切片资料源与采集客户端

此仓库提供本地授权演练用来源服务、协议样本、来源自身的测试，以及对接该来源的采集客户端 `collector/`。全部机构、课程、账号、作品及观测均为合成资料。使用前阅读 [来源契约](docs/来源契约.md) 与 [采集工具说明](docs/采集工具.md)。

Python 3.11及以上。安装固定依赖后，运行：

    python -m pip install -r requirements.txt
    python -m source.server --port 8765

服务默认绑定本机环境的8765端口，健康检查地址为http://127.0.0.1:8765/health。浏览器门户入口为契约所列资料端点。服务重启会重置演练状态。

## 采集客户端

按物理坐标选择资料，自动分页、保留每切片版次与坐标、区分缺测与质控不可用单元，
并在版次退休/轴中间补点时按坐标重建且不产生混版“完整”阵列。数据目录与代码目录分离，
建议用绝对脚本路径配合隔离模式运行（当前目录不参与导入）：

    python -I $(pwd)/collect_data.py --base http://127.0.0.1:8765 collect \
        --workdir ./data/job1 --select "time=0:5,station=10:30,sample=0:4.6"
    python -I $(pwd)/run_drills.py --data-root ./data/drill-data

详见 [采集工具说明](docs/采集工具.md)。

## 测试

来源与采集客户端测试：

    python -m unittest discover -s tests -v

独立容器可用docker build -t local-source .构建，docker run --rm -p 8765:8765 local-source启动。来源测试也可在构建镜像中运行：docker run --rm --network none local-source python -m unittest discover -s tests -v。容器使用非root用户，测试仅连接同容器回环源；笔记本WebSocket也在同容器内连接。

测试只检查来源如何发布资料与响应页面操作，不能证明收集、离线交付或业务解释已经实现。控制接口仅供本地演练。文档与代码没有真实服务凭据。
