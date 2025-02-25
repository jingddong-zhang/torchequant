import logging

# 创建Logger对象
logger = logging.getLogger(__name__)

# 设置日志级别为INFO，只记录INFO级别及以上的日志
logger.setLevel(logging.INFO)

# 创建一个文件处理器，将日志输出到名为 'train_log.txt' 的文件中
log_filename = 'train_log.txt'
file_handler = logging.FileHandler(log_filename)

# 创建一个日志格式化器，用于指定日志的输出格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将文件处理器添加到Logger中
logger.addHandler(file_handler)

# 训练过程中记录日志
for epoch in range(100):
    # 模型训练代码...
    # 记录训练结果
    train_loss = 0.0  # 假设这是你的训练损失
    logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

# 关闭文件处理器
file_handler.close()