from django.apps import AppConfig
import threading
import time

# 避免循环导入问题
_entity_linker_initialized = threading.Event()

class XyNeo4jConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'xy_neo4j'
    
    def ready(self):
        """应用启动时执行的代码"""
        # 防止在开发环境中reload导致多次初始化
        if not _entity_linker_initialized.is_set():
            # 使用后台线程初始化EntityLinker
            # 这样不会阻塞Django的启动过程
            thread = threading.Thread(target=self._init_entity_linker)
            thread.daemon = True
            thread.start()
            
            # 另一个线程负责定期刷新缓存
            refresh_thread = threading.Thread(target=self._periodic_refresh)
            refresh_thread.daemon = True
            refresh_thread.start()
    
    def _init_entity_linker(self):
        """在单独线程中初始化EntityLinker"""
        try:
            # 导入放在方法内部避免循环导入
            from .Entity_Mention.Entity_Order import EntityLinker
            
            print("开始预初始化EntityLinker单例...")
            # 获取实例会触发初始化
            EntityLinker.get_instance()
            _entity_linker_initialized.set()
            print("EntityLinker单例预初始化完成")
        except Exception as e:
            print(f"EntityLinker预初始化失败: {e}")
    
    def _periodic_refresh(self):
        """定期刷新EntityLinker缓存"""
        try:
            # 首次等待初始化完成
            _entity_linker_initialized.wait(timeout=300)
            
            from .Entity_Mention.Entity_Order import EntityLinker
            
            # 每24小时刷新一次
            while True:
                time.sleep(86400)  # 24小时
                try:
                    entity_linker = EntityLinker.get_instance()
                    entity_linker.check_and_refresh_if_needed(force=True)
                    print("完成EntityLinker定期缓存刷新")
                except Exception as e:
                    print(f"EntityLinker定期刷新失败: {e}")
        except Exception as e:
            print(f"启动定期刷新线程失败: {e}") 