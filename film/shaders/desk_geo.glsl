// The room, as distance functions. x right, y up, z out of the screen toward the operator; metres.
// The glass is the plane z=0. Everything that receives light is here, for the bake and for the camera.
const float HW=.2985,HH=.168;              // half the panel (SCREEN_W, SCREEN_H)
const float DY=-.232;                      // desktop
const vec3 RMIN=vec3(-1.05,-.952,-.40),RMAX=vec3(1.35,1.38,2.00);
const vec4 WINDOW=vec4(-.72,-.12,.42,1.32); // x0,y0,x1,y1 on the wall behind the operator
const vec2 LED_P=vec2(.300,-.2005);

const int M_WALL=1,M_DESK=2,M_BEZEL=3,M_STAND=5,M_MACH=6,M_KEY=7,M_KCASE=8,M_MUG=9,M_PAPER=10,M_SPK=11,M_CONE=12,
          M_CURT=13,M_SHELF=14,M_BOOK=15,M_CABLE=16,M_PLATE=17,M_COFFEE=18,M_PEN=19,M_FLOOR=20,M_CEIL=21,M_SIDE=22;
const int G_ROOM=0,G_DESK=1,G_MON=2,G_STAND=3,G_MACH=4,G_KB=5,G_MUG=6,G_PAPER=7,G_SPK=8,G_CURT=9,G_SHELF=10,G_CABLE=11,G_PLATE=12;

int groupOf(int m){
  if(m==M_WALL||m==M_FLOOR||m==M_CEIL||m==M_SIDE) return G_ROOM;
  if(m==M_DESK) return G_DESK;
  if(m==M_BEZEL) return G_MON;
  if(m==M_STAND) return G_STAND;
  if(m==M_MACH) return G_MACH;
  if(m==M_KEY||m==M_KCASE) return G_KB;
  if(m==M_MUG||m==M_COFFEE) return G_MUG;
  if(m==M_PAPER||m==M_PEN) return G_PAPER;
  if(m==M_SPK||m==M_CONE) return G_SPK;
  if(m==M_CURT) return G_CURT;
  if(m==M_SHELF||m==M_BOOK) return G_SHELF;
  if(m==M_CABLE) return G_CABLE;
  return G_PLATE;
}

// what one bounce of light sees: flat, measured colours. the camera's materials live in the scene pass
vec3 bounceAlbedo(int m){
  if(m==M_WALL)  return vec3(.60,.58,.54);
  if(m==M_SIDE)  return vec3(.34,.33,.31);
  if(m==M_FLOOR) return vec3(.11,.085,.06);
  if(m==M_CEIL)  return vec3(.46,.46,.44);
  if(m==M_DESK)  return vec3(.33,.21,.11);
  if(m==M_BEZEL) return vec3(.020);
  if(m==M_STAND) return vec3(.045);
  if(m==M_MACH)  return vec3(.035);
  if(m==M_KEY)   return vec3(.045);
  if(m==M_KCASE) return vec3(.030);
  if(m==M_MUG)   return vec3(.09,.09,.10);
  if(m==M_COFFEE)return vec3(.02,.012,.008);
  if(m==M_PAPER) return vec3(.78,.77,.72);
  if(m==M_PEN)   return vec3(.03);
  if(m==M_SPK)   return vec3(.022);
  if(m==M_CONE)  return vec3(.028);
  if(m==M_CURT)  return vec3(.30,.31,.32);
  if(m==M_SHELF) return vec3(.52,.50,.45);
  if(m==M_BOOK)  return vec3(.20,.17,.15);
  if(m==M_CABLE) return vec3(.014);
  if(m==M_PLATE) return vec3(.44,.43,.40);
  return vec3(.3);
}

uniform int uMask;                         // which groups exist (the bake captures them one at a time)
float gD; int gM;
void U(float d,int m){ if(d<gD){ gD=d; gM=m; } }

float sdBox(vec3 p,vec3 b){ vec3 q=abs(p)-b; return length(max(q,0.))+min(max(q.x,max(q.y,q.z)),0.); }
float sdBox2(vec2 p,vec2 b){ vec2 q=abs(p)-b; return length(max(q,0.))+min(max(q.x,q.y),0.); }
float sdRBox(vec3 p,vec3 b,float r){ return sdBox(p,b-r)-r; }
float smin(float a,float b,float k){ float h=clamp(.5+.5*(b-a)/k,0.,1.); return mix(b,a,h)-k*h*(1.-h); }
float ext(float d2,float z,float h,float r){ vec2 w=vec2(d2+r,abs(z)-h+r); return min(max(w.x,w.y),0.)+length(max(w,0.))-r; }
float seg(vec3 p,vec3 a,vec3 b){ vec3 pa=p-a,ba=b-a; return length(pa-ba*clamp(dot(pa,ba)/dot(ba,ba),0.,1.)); }
float hash1(float x){ x=fract(x*.1031); x*=x+33.33; x*=x+x; return fract(x); }

const int ON_ALL=8191;

void addDesk(vec3 p){
  float b=sdRBox(p-vec3(0.,DY-.014,.08),vec3(.80,.014,.38),.0018);
  U(b,M_DESK);
}

void addMonitor(vec3 p){
  float bb=sdBox(p-vec3(0.,-.0035,-.0335),vec3(.3085,.1812,.0365));
  if(bb>.02){ U(bb,M_BEZEL); return; }
  // the frame: thin plastic around the panel, a heavier chin
  vec2 q=p.xy-vec2(0.,-.00325);
  float dr=sdBox2(q,vec2(.307,.17975)-.0055)-.0055;
  float frame=ext(dr,p.z+.0071,.0089,.0017);
  // panel sits behind a chamfered lip
  vec2 qs=abs(p.xy)-vec2(HW+.0004,HH+.0004);
  float lip=max(qs.x,qs.y)-max(p.z,0.)*.85;
  float rec=max(lip,-p.z);
  float d=max(frame,-rec);
  // a seam where the shell meets the frame, on the outside
  d=max(d,-max(abs(p.z+.0125)-.00035,-(dr+.0009)));
  // the rear housing: only ever seen as what blocks light
  float back=sdRBox(p-vec3(0.,-.015,-.040),vec3(.24,.135,.030),.022);
  d=smin(d,back,.016);
  U(d,M_BEZEL);
}

void addStand(vec3 p){
  float bb=sdBox(p-vec3(0.,-.1345,-.058),vec3(.122,.0985,.122));
  if(bb>.02){ U(bb,M_STAND); return; }
  vec3 q=p-vec3(0.,-.135,-.085); q.z+=q.y*.11;
  float f=clamp((.10-q.y)/.20,0.,1.);       // 0 at the housing, 1 at the foot
  float hw=.0235+.009*f*f*f;
  float neck=sdRBox(q,vec3(hw,.099,.0098+.004*f*f),.0072);
  vec3 g=p-vec3(0.,-.2275,-.058);
  float dr=sdBox2(g.xz,vec2(.118,.118)-.032)-.032;
  float foot=ext(dr,g.y+g.z*.010,.0046,.0022);
  U(smin(neck,foot,.012),M_STAND);
}

void addMachine(vec3 p){
  vec3 q=p-vec3(.30,-.2065,-.125);
  float bb=sdBox(q,vec3(.074,.0265,.074));
  if(bb>.015){ U(bb,M_MACH); return; }
  float dr=sdBox2(q.xz,vec2(.072)-.017)-.017;
  float d=ext(dr,q.y,.0245,.0024);
  d=max(d,-max(abs(q.y+.0212)-.0004,-(dr+.0008)));  // the seam above the base plate
  U(d,M_MACH);
}

const float KW[96]=float[96](
 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,2.,1.,0.,
 1.5,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.5,1.,0.,
 1.75,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,2.25,1.,0.,0.,
 2.25,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.75,1.,1.,0.,0.,
 1.25,1.25,1.25,6.25,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.);
const float KU=.01905;

vec3 kbLocal(vec3 p){
  vec3 q=p-vec3(-.138,DY,.185);
  const float c=.99875,s=.04998;           // not quite square to the monitor
  q.xz=vec2(c*q.x-s*q.z,s*q.x+c*q.z);
  return q;
}
// key under local xz: x = key centre, y = half width, z = row
vec3 kbCell(vec3 q){
  float zr=(q.z+.0572)/KU;
  float row=clamp(floor(zr),0.,5.);
  float x=(q.x+.1524)/KU;
  float e=0.,w=1.;
  int r=int(row);
  for(int i=0;i<16;i++){
    w=KW[r*16+i];
    if(w==0.){ w=1.; break; }
    if(x<e+w||i==15) break;
    e+=w;
  }
  return vec3((e+w*.5)*KU-.1524,w*.5*KU,row);
}
void addKeyboard(vec3 p){
  vec3 q=kbLocal(p);
  float bb=sdBox(q-vec3(0.,.0185,0.),vec3(.164,.0205,.074));
  if(bb>.012){ U(bb,M_KCASE); return; }
  q.y-=-q.z*.07;                             // a few degrees of wedge
  float cs=sdRBox(q-vec3(0.,.0125,0.),vec3(.1605,.0135,.0695),.0035);
  // sunk key deck
  cs=max(cs,-sdBox(q-vec3(0.,.03,0.),vec3(.1535,.0045,.0585)));
  U(cs,M_KCASE);
  vec3 kc=kbCell(q);
  float cz=(kc.z+.5)*KU-.0572;
  vec3 k=q-vec3(kc.x,0.,cz);
  float h=(k.y-.0325)/.0055;                 // -1 at the plate, 1 at the top
  float taper=(h*.5+.5)*.0016;
  float dish=(1.-pow(k.x/max(kc.y,1e-3),2.))*.00065;
  float kd=sdRBox(k-vec3(0.,.0325-dish*step(0.,h),0.),vec3(kc.y-.0009-taper,.0055,.0086-taper),.0013);
  U(kd,M_KEY);
}

void addMug(vec3 p){
  vec3 q=p-vec3(.17,DY,.378);
  float bb=sdBox(q-vec3(.012,.0445,0.),vec3(.062,.0465,.0465));
  if(bb>.012){ U(bb,M_MUG); return; }
  float r=length(q.xz);
  vec2 a=vec2(r,q.y-.044);
  float outer=sdBox2(a,vec2(.0415,.044)-.0035)-.0035;
  vec2 ai=vec2(r,q.y-.0545);
  float inner=sdBox2(ai,vec2(.0382,.0515)-.003)-.003;
  float d=max(outer,-inner);
  // the handle
  vec3 h=q-vec3(.0425,.046,0.);
  float t=length(vec2(length(vec2(h.x*.92,h.y))-.0225,h.z*.72))-.0052;
  t=max(t,.0395-r);
  d=smin(d,t,.003);
  U(d,M_MUG);
  U(max(abs(q.y-.0705)-.0004,r-.0385),M_COFFEE);
}

void addPaper(vec3 p){
  vec3 q=p-vec3(-.262,DY,.042);
  float bb=sdBox(q-vec3(0.,.008,0.),vec3(.175,.012,.16));
  if(bb>.012){ U(bb,M_PAPER); return; }
  // a loose A4 sheet, its corner lifting
  vec2 a=vec2(.9744*q.x-.2250*q.z,.2250*q.x+.9744*q.z);
  float lift=.0028*smoothstep(.03,.15,a.x-a.y*.5)+.0004*sin(a.y*38.);
  float s=sdBox(vec3(a.x,q.y-.0007-lift,a.y),vec3(.105,.00028,.148));
  U(s*.8,M_PAPER);
  // a notepad on top of it
  vec2 b=vec2(.9877*(q.x-.012)+.1564*(q.z+.012),-.1564*(q.x-.012)+.9877*(q.z+.012));
  float n=sdBox(vec3(b.x,q.y-.0042,b.y),vec3(.074,.0032,.102))-.0003;
  U(n,M_PAPER);
  // and a pen across it
  vec3 pa=vec3(.043,.0102,-.05),pb=vec3(-.052,.0102,.062);
  float pn=seg(q,pa,pb)-.0043;
  U(pn,M_PEN);
}

void addSpeaker(vec3 p){
  vec3 q=p-vec3(-.402,DY+.105,-.175);
  float bb=sdBox(q,vec3(.073,.106,.081));
  if(bb>.015){ U(bb,M_SPK); return; }
  float d=sdRBox(q,vec3(.071,.105,.079),.0055);
  vec2 f=q.xy-vec2(0.,-.032);
  float rf=length(f);
  float bowl=length(vec3(f,q.z-.079-.066))-.083;  // the cone: a shallow dish
  float carve=max(bowl,rf-.051);
  d=max(d,-carve);
  // the surround and the dust cap
  float sur=length(vec2(rf-.052,q.z-.0795))-.0042;
  d=min(d,sur);
  float cap=length(vec3(f,q.z-.057))-.017;
  d=min(d,cap);
  // a tweeter, small and proud
  vec2 tp=q.xy-vec2(0.,.064);
  float tw=length(vec3(tp,q.z-.0735))-.0125;
  d=min(d,tw);
  float ring=length(vec2(length(tp)-.0165,q.z-.0795))-.0024;
  d=min(d,ring);
  int m=(rf<.0515&&q.z>.03)||(length(tp)<.0135&&q.z>.03)?M_CONE:M_SPK;
  U(d,m);
}

void addCurtain(vec3 p){
  if(p.x<.36){ U(.40-p.x,M_CURT); return; }
  float x=p.x-.432-.005*sin(p.y*2.3+.6);
  float w=.015*sin(x*48.+1.3*sin(p.y*1.1)+.7)+.008*sin(x*111.+2.1*sin(p.y*1.9+x*9.)+1.4);
  float d=abs(p.z+.345-w)-.002;
  d=max(d,-x);
  d=max(d,p.y-1.50);
  U(d*.62,M_CURT);
}

void addShelf(vec3 p){
  float bb=sdBox(p-vec3(-.07,.385,-.31),vec3(.64,.135,.095));
  if(bb>.02){ U(bb,M_SHELF); return; }
  float d=sdRBox(p-vec3(-.07,.265,-.31),vec3(.62,.0105,.09),.0014);
  // two steel brackets under it
  vec3 q=vec3(abs(p.x+.07)-.40,p.y-.242,p.z+.335);
  d=min(d,sdBox(q,vec3(.004,.013,.052)));
  U(d,M_SHELF);
  // books standing on it
  float x=p.x+.56;
  float id=clamp(floor(x/.0345),0.,21.);
  float hh=.083+.05*hash1(id+3.)-(id>12.&&id<15.?.2:0.);
  float w=.0118+.0035*hash1(id+11.);
  float dep=.05+.012*hash1(id+5.);
  vec3 c=vec3(p.x+.56-(id+.5)*.0345,p.y-(.2755+hh),p.z+.31);
  float bk=sdRBox(c,vec3(w,hh,dep),.0012);
  U(bk,M_BOOK);
}

void addCables(vec3 p){
  float bb=sdBox(p-vec3(.03,-.20,-.195),vec3(.33,.11,.155));
  if(bb>.012){ U(bb,M_CABLE); return; }
  const float r=.0026;
  float d=seg(p,vec3(.052,-.11,-.078),vec3(.062,-.185,-.092));
  d=min(d,seg(p,vec3(.062,-.185,-.092),vec3(.083,-.2225,-.118)));
  d=min(d,seg(p,vec3(.083,-.2225,-.118),vec3(.098,-.2292,-.165)));
  d=min(d,seg(p,vec3(.098,-.2292,-.165),vec3(.150,-.2292,-.245)));
  d=min(d,seg(p,vec3(.150,-.2292,-.245),vec3(.170,-.2292,-.310)));
  d=min(d,seg(p,vec3(.170,-.2292,-.310),vec3(.176,-.33,-.345)));
  // from the machine, thinner, across the back of the desk
  float e=seg(p,vec3(.228,-.222,-.17),vec3(.17,-.2298,-.20));
  e=min(e,seg(p,vec3(.17,-.2298,-.20),vec3(.04,-.2298,-.215)));
  e=min(e,seg(p,vec3(.04,-.2298,-.215),vec3(-.05,-.2298,-.19)));
  e=min(e,seg(p,vec3(-.05,-.2298,-.19),vec3(-.046,-.17,-.10)));
  d=min(d-r,e-.0018);
  // and to the speaker
  float g=seg(p,vec3(-.331,-.205,-.20),vec3(-.24,-.2300,-.245));
  g=min(g,seg(p,vec3(-.24,-.2300,-.245),vec3(-.10,-.2300,-.28)));
  g=min(g,seg(p,vec3(-.10,-.2300,-.28),vec3(-.04,-.2300,-.315)));
  d=min(d,g-.0015);
  U(d,M_CABLE);
}

void addPlate(vec3 p){
  float bb=sdBox(p-vec3(-.468,.058,-.396),vec3(.046,.046,.006));
  if(bb>.01){ U(bb,M_PLATE); return; }
  U(sdRBox(p-vec3(-.468,.058,-.397),vec3(.043,.043,.004),.0018),M_PLATE);
}

// the distance to everything but the room's own six planes
float map(vec3 p){
  gD=1e3; gM=0;
  if((uMask&(1<<G_DESK))!=0) addDesk(p);
  if((uMask&(1<<G_MON))!=0) addMonitor(p);
  if((uMask&(1<<G_STAND))!=0) addStand(p);
  if((uMask&(1<<G_MACH))!=0) addMachine(p);
  if((uMask&(1<<G_KB))!=0) addKeyboard(p);
  if((uMask&(1<<G_MUG))!=0) addMug(p);
  if((uMask&(1<<G_PAPER))!=0) addPaper(p);
  if((uMask&(1<<G_SPK))!=0) addSpeaker(p);
  if((uMask&(1<<G_CURT))!=0) addCurtain(p);
  if((uMask&(1<<G_SHELF))!=0) addShelf(p);
  if((uMask&(1<<G_CABLE))!=0) addCables(p);
  if((uMask&(1<<G_PLATE))!=0) addPlate(p);
  return gD;
}

vec3 mapNormal(vec3 p,float e){
  const vec2 k=vec2(1.,-1.);
  return normalize(k.xyy*map(p+k.xyy*e)+k.yyx*map(p+k.yyx*e)+k.yxy*map(p+k.yxy*e)+k.xxx*map(p+k.xxx*e));
}

// distance along a ray to the inside of the room, and which plane it meets
float roomExit(vec3 ro,vec3 rd,out vec3 n,out int m){
  vec3 inv=1./rd;
  vec3 t=vec3(rd.x>0.?RMAX.x:RMIN.x,rd.y>0.?RMAX.y:RMIN.y,rd.z>0.?RMAX.z:RMIN.z);
  t=(t-ro)*inv;
  if(t.x<t.y&&t.x<t.z){ n=vec3(-sign(rd.x),0.,0.); m=M_SIDE; return t.x; }
  if(t.z<t.y){ n=vec3(0.,0.,-sign(rd.z)); m=rd.z<0.?M_WALL:M_SIDE; return t.z; }
  n=vec3(0.,-sign(rd.y),0.); m=rd.y>0.?M_CEIL:M_FLOOR; return t.y;
}

// sphere-trace to whatever a ray meets. false only when it left through tmax first
bool traceRoom(vec3 ro,vec3 rd,float tmax,int steps,out float t,out int m,out vec3 n){
  vec3 rn; int rm;
  float te=roomExit(ro,rd,rn,rm);
  float lim=min(te,tmax);
  t=0.;
  for(int i=0;i<steps;i++){
    float d=map(ro+rd*t);
    if(d<.00025*(1.+t)){
      m=gM; n=mapNormal(ro+rd*t,.00035);
      return true;
    }
    t+=d;
    if(t>=lim) break;
  }
  if(t>=te||te<=tmax){ t=te; m=rm; n=rn; return true; }
  return false;
}
