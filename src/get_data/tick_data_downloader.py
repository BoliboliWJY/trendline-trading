def download_tick_data(backtest_tick_path, coin_type, contract_type, start_time, end_time, max_retries=3, retry_delay=5):
    import os
    import requests
    import hashlib
    import zipfile
    import time
    from datetime import datetime, timedelta
    from requests.exceptions import ConnectionError, ChunkedEncodingError, ReadTimeout
    from urllib3.exceptions import IncompleteRead, ProtocolError
    
    # 确保存储路径存在
    os.makedirs(backtest_tick_path, exist_ok=True)
    
    # 将时间字符串转换为datetime对象
    start_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_date = datetime.strptime(end_time, '%Y-%m-%d')
    
    current_date = start_date
    
    # 请求头，模拟浏览器
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 在指定的时间范围内循环
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # 构建URL - 修正为直接下载路径
        base_url = f"https://data.binance.vision/data/futures/{contract_type}/daily/trades/{coin_type}/"
        file_name = f"{coin_type}-trades-{date_str}.zip"
        checksum_file_name = f"{file_name}.CHECKSUM"
        
        file_url = base_url + file_name
        checksum_url = base_url + checksum_file_name
        
        local_file_path = os.path.join(backtest_tick_path, file_name)
        local_checksum_path = os.path.join(backtest_tick_path, checksum_file_name)
        
        # 检查文件是否已存在，如果存在则跳过
        if os.path.exists(local_file_path):
            print(f"文件已存在，跳过下载: {local_file_path}")
            # 移至下一天
            current_date += timedelta(days=1)
            continue
            
        print(f"准备下载: {file_url}")
        
        # 重试计数器
        retry_count = 0
        download_success = False
        
        # 下载ZIP文件（带重试）
        while retry_count < max_retries and not download_success:
            if retry_count > 0:
                print(f"第 {retry_count} 次重试下载 {date_str}...")
                time.sleep(retry_delay)  # 重试前等待
            
            try:
                # 获取文件大小
                head_response = requests.head(file_url, headers=headers, timeout=10)
                if head_response.status_code != 200:
                    print(f"无法获取文件信息 {date_str}: HTTP {head_response.status_code}")
                    break
                
                total_size = int(head_response.headers.get('content-length', 0))
                
                # 检查文件是否已经存在且大小正确
                if os.path.exists(local_file_path) and os.path.getsize(local_file_path) == total_size:
                    print(f"文件已存在且大小正确: {local_file_path}")
                    download_success = True
                    break
                
                # 如果文件已存在但不完整，则继续下载
                downloaded = 0
                mode = 'wb'
                headers_copy = headers.copy()
                
                if os.path.exists(local_file_path):
                    downloaded = os.path.getsize(local_file_path)
                    if downloaded < total_size:
                        mode = 'ab'
                        headers_copy['Range'] = f'bytes={downloaded}-'
                        print(f"继续下载文件: 已下载 {downloaded/1024/1024:.2f} MB / 共 {total_size/1024/1024:.2f} MB")
                    else:
                        # 文件可能已完成，但我们不确定其完整性，重新下载
                        mode = 'wb'
                        downloaded = 0
                
                print(f"下载数据: {date_str} (大小: {total_size/1024/1024:.2f} MB)")
                
                # 使用更长的超时时间
                response = requests.get(file_url, headers=headers_copy, stream=True, timeout=30)
                
                if response.status_code in [200, 206]:
                    with open(local_file_path, mode) as f:
                        for chunk in response.iter_content(chunk_size=1024*1024):  # 使用1MB块
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                # 每10MB输出一次进度
                                if downloaded % (10*1024*1024) < 1024*1024:
                                    print(f"进度: {downloaded/total_size*100:.1f}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)")
                    
                    print(f"文件下载完成: {local_file_path}")
                    download_success = True
                else:
                    print(f"无法下载数据文件 {date_str}: HTTP {response.status_code}")
                    print(f"响应内容: {response.text[:200]}")
                    retry_count += 1
            
            except (ConnectionError, ChunkedEncodingError, ReadTimeout, IncompleteRead, ProtocolError) as e:
                print(f"下载过程中发生连接错误: {type(e).__name__}: {str(e)}")
                retry_count += 1
                if os.path.exists(local_file_path):
                    print("文件已部分下载，下次重试将继续下载")
            
            except Exception as e:
                print(f"下载 {date_str} 时出错: {type(e).__name__}: {str(e)}")
                retry_count += 1
        
        # 如果下载成功，继续校验文件
        if download_success:
            try:
                # 下载校验和文件（带重试）
                checksum_retry_count = 0
                checksum_success = False
                
                while checksum_retry_count < max_retries and not checksum_success:
                    if checksum_retry_count > 0:
                        print(f"第 {checksum_retry_count} 次重试下载校验和文件...")
                        time.sleep(retry_delay)
                    
                    try:
                        checksum_response = requests.get(checksum_url, headers=headers, timeout=10)
                        if checksum_response.status_code == 200:
                            with open(local_checksum_path, 'wb') as f:
                                f.write(checksum_response.content)
                            
                            print(f"校验和文件下载完成: {local_checksum_path}")
                            checksum_success = True
                        else:
                            print(f"无法下载校验和文件 {date_str}: HTTP {checksum_response.status_code}")
                            checksum_retry_count += 1
                    
                    except Exception as e:
                        print(f"下载校验和文件时出错: {type(e).__name__}: {str(e)}")
                        checksum_retry_count += 1
                
                # 如果校验和文件下载成功，验证文件完整性
                if checksum_success:
                    with open(local_checksum_path, 'r') as f:
                        checksum_content = f.read().strip()
                        expected_checksum = checksum_content.split()[0]
                    
                    print(f"期望的校验和: {expected_checksum}")
                    
                    # 计算下载文件的SHA256校验和
                    print("正在计算文件校验和...")
                    sha256_hash = hashlib.sha256()
                    with open(local_file_path, 'rb') as f:
                        for byte_block in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(byte_block)
                    actual_checksum = sha256_hash.hexdigest()
                    
                    print(f"计算得到的校验和: {actual_checksum}")
                    
                    if expected_checksum == actual_checksum:
                        print(f"校验成功: {date_str}")
                        
                        # 可选：解压文件
                        # with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                        #     zip_ref.extractall(os.path.join(backtest_tick_path, f"{date_str}"))
                        
                        # 校验成功后删除checksum文件
                        os.remove(local_checksum_path)
                        print(f"已删除校验和文件: {local_checksum_path}")
                    else:
                        print(f"校验失败！{date_str}")
                        print(f"预期: {expected_checksum}")
                        print(f"实际: {actual_checksum}")
                        # 删除损坏的文件
                        print(f"删除损坏的文件...")
                        os.remove(local_file_path)
                        os.remove(local_checksum_path)
                else:
                    print(f"无法下载校验和文件，跳过校验")
                    if os.path.exists(local_checksum_path):
                        os.remove(local_checksum_path)
            
            except Exception as e:
                print(f"处理校验和时出错: {type(e).__name__}: {str(e)}")
                # 清理可能存在的校验和文件
                if os.path.exists(local_checksum_path):
                    os.remove(local_checksum_path)
        
        else:
            print(f"在 {max_retries} 次尝试后仍无法下载 {date_str}，跳过此日期")
            # 删除可能存在的部分下载文件
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
        
        # 移至下一天
        current_date += timedelta(days=1)
    
    print("数据下载完成")
    return True

if __name__ == "__main__":
    contract_type = "um"
    coin_type = "SOLUSDT"
    backtest_tick_path = f"backtest/{contract_type}/tick/{coin_type}"
    download_tick_data(backtest_tick_path, coin_type, contract_type, "2025-02-01", "2025-03-30")
