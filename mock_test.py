import sys, json, random, asyncio
sys.path.insert(0, ".")
import gem, config as C
random.seed(0)
async def fake(prompt, *, model=None, images=(), schema=None, tools=None, video_url=None, temperature=0, tag="", sample=0):
    n = getattr(schema, "__name__", None)
    if n == "FrameLabels":
        return {"items": [{"key": k, "present": random.random()<.5, "confidence": random.random(), "box_2d": [100,100,400,400], "evidence": "x"} for k in C.LABELS]}
    if n == "CropID": return {"answer": random.choice([v[1] for v in C.LABELS.values()]), "confidence": .8}
    if n == "Critique": return {"rebuttals": [{"key": k, "argument": "maybe", "p_wrong": random.random()} for k in C.LABELS]}
    if n == "Verdict": return {"present": random.random()<.5, "confidence": random.random(), "reasoning": "looked"}
    if n == "Manual": return {"procedure":"x","phases":["a"],"structures":[{"key":k,"name":k,"appearance":"red","where_found":"c","when_visible":"a","confused_with":["y"],"how_to_tell_apart":"z"} for k in C.LABELS],"critical_events":["cvs"],"general_rules":["r"]}
    if n == "VideoLesson": return {"rules":["r"],"moments":["00:10 x"]}
    if n == "Line": return {"narration":"Gallbladder retracted.","unsure_about":[]}
    if n == "Timeline": return {"phases":[{"name":"a","start":"00:00","end":"01:00","evidence":"e","confidence":.9}],"events":[{"time":"00:30","event":"cvs","evidence":"e","confidence":.7}],"sightings":[{"time":"00:10","key":"liver","confidence":.9}]}
    if n == "Audit": return {"problems":[],"unverified_items":[],"overall_confidence":.8}
    return "research notes"
gem.call = fake
step = sys.argv[1]
if step == "med":
    import med_school; sys.argv=["x"]; med_school.gem.call = fake; med_school.main()
elif step == "label":
    import label; label.gem.call = fake
    asyncio.run(label.run(sys.argv[2], sys.argv[3].split(","), int(sys.argv[4]), 0))
elif step == "accept":
    import accept; accept.gem.call = fake; asyncio.run(accept.main(True, 300))
elif step == "narrate":
    import narrate; narrate.gem.call = fake; asyncio.run(narrate.main(sys.argv[2]))
elif step == "atlas":
    import atlas, med_school; atlas.gem.call = fake; med_school.gem.call = fake
    asyncio.run(atlas.main("laparoscopic appendectomy", "https://youtube.com/watch?v=x"))
