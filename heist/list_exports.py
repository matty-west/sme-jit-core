import frida
import sys
import time

def on_message(message, data):
    print(message)

def main():
    try:
        pid = frida.spawn(["./heist/amx_runner"])
        session = frida.attach(pid)
        
        # This script just finds and lists sgemm-related exports
        script = session.create_script("""
            const modules = Process.enumerateModules();
            modules.forEach(m => {
                if (m.name.toLowerCase().includes('blas') || 
                    m.name.toLowerCase().includes('accelerate') ||
                    m.name.toLowerCase().includes('veclib')) {
                    console.log('[*] Module: ' + m.name + ' @ ' + m.base);
                    m.enumerateExports().forEach(exp => {
                        if (exp.name.includes('sgemm')) {
                            console.log('    Export: ' + exp.name + ' @ ' + exp.address);
                        }
                    });
                }
            });
        """)
        script.on('message', on_message)
        script.load()
        frida.resume(pid)
        time.sleep(2)
        session.detach()
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == '__main__':
    main()
