import frida
import sys
import time

JS_CODE = """
'use strict';

console.log("[*] Nuclear Trace active");

function followThread(threadId) {
    console.log("[*] Following thread " + threadId);
    Stalker.follow(threadId, {
        transform: function(iterator) {
            let instruction;
            while ((instruction = iterator.next()) !== null) {
                const op = instruction.address.readU32() >>> 0;
                if ((op & 0xFFF00000) === 0x00200000) {
                    const xn = op & 0x1F;
                    iterator.putCallout(function(context) {
                        const val = context['x' + xn].toString();
                        send({ type: 'amx_val', addr: instruction.address, op: op, xn: xn, val: val, tid: threadId });
                    });
                }
                iterator.keep();
            }
        }
    });
}

// Follow any thread that is currently running
Process.enumerateThreads().forEach(t => followThread(t.id));

// Follow any new thread that gets created
/* 
// Stalker.onThreadCreated is not a function, 
// we have to hook thread creation or just periodic polling
*/
setInterval(() => {
    Process.enumerateThreads().forEach(t => {
        // This is inefficient but will catch them
        try {
            followThread(t.id);
        } catch (e) {} // already following
    });
}, 500);
"""

amx_vals = []

def on_message(message, data):
    if message['type'] == 'send':
        payload = message['payload']
        if payload['type'] == 'amx_val':
            print(f"[*] AMX [TID {payload['tid']}]: {payload['op']:08x} x{payload['xn']}={payload['val']}")
            amx_vals.append(payload)
    else:
        print(message)

def main():
    try:
        pid = frida.spawn(["./heist/amx_runner"])
        session = frida.attach(pid)
        script = session.create_script(JS_CODE)
        script.on('message', on_message)
        script.load()
        frida.resume(pid)
        
        print("[*] Waiting for AMX on any thread...")
        time.sleep(15)
            
        if amx_vals:
            with open('heist_amx_vals.json', 'w') as f:
                import json
                json.dump(amx_vals, f, indent=2)
            print(f"[*] Saved {len(amx_vals)} values to heist_amx_vals.json")
        else:
            print("[!] No AMX values captured on any thread.")
        sys.exit(0)
            
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == '__main__':
    main()
