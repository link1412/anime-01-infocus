#version 410
// The camera's look at the room. The glass is a plane with a real pixel grid; everything else is found by rays and
// shaded with the baked light (a weighted sum of passes, weights from what the screen shows now) plus view-dependent gloss.
#include "common.glsl"
in vec2 uv;
layout(location=0) out vec4 o;       // the glass: emitted light, premultiplied by coverage
layout(location=1) out vec4 oRoom;   // the room: rgb, a = 1/depth (0 where only the glass is)
uniform sampler2D uScreen,uDrive,uA0,uA1,uA2,uA3,uAW;
uniform vec3 uO,uF,uR,uU;
uniform vec2 uRes;
uniform float uTan,uAspect,uW,uH,uBright,uMoire,uIsFeed,uT,uGrid,uGlow,uLed;
uniform vec4 uCardRect[40],uCardDom[40],uCardPlane[40];
uniform int uCardOf[96];
#include "desk_geo.glsl"

vec3 gC[4]; vec3 gCW;
// a real room has more in it to bounce the light around than this one: curtains, a door, a bed
const float ROOM_SPILL=1.35;
const vec3 WIN_TINT=vec3(.62,.70,.90)*.022;

float boxd(vec2 p,vec2 b){ vec2 d=abs(p)-b; return length(max(d,0.))+min(max(d.x,d.y),0.); }

// ---- light: baked passes, weighted by the present frame
vec3 lightAt(vec3 p,vec3 n,int g){
  vec3 sum=vec3(0.); float ws=0.;
  vec3 n2=n*n;
  for(int a=0;a<3;a++){
    if(n2[a]<.015) continue;
    int c=uCardOf[g*6+a*2+(n[a]>0.?0:1)];
    if(c<0) continue;
    vec2 q=a==0?p.yz:(a==1?p.xz:p.xy);
    vec4 D=uCardDom[c];
    vec2 st=(q-D.xy)/D.zw;
    if(st.x<0.||st.y<0.||st.x>1.||st.y>1.) continue;
    vec4 P=uCardPlane[c];
    vec4 Rc=uCardRect[c];
    vec2 at=Rc.xy+Rc.zw*st;
    vec4 a0=texture(uA0,at);
    float dp=(P.x-p[a])*P.y;
    float w=n2[a]*(1.-smoothstep(.003,.012,abs(a0.w-dp)));
    if(w<1e-4) continue;
    sum+=w*(a0.rgb*gC[0]+texture(uA1,at).rgb*gC[1]+texture(uA2,at).rgb*gC[2]+texture(uA3,at).rgb*gC[3]+texture(uAW,at).rgb*gCW);
    ws+=w;
  }
  if(ws<.02) return (gC[0]+gC[1]+gC[2]+gC[3])*.0016+gCW*.004;
  return sum/ws;
}

// ---- the screen as a light source seen in gloss
vec3 screenAt(vec2 s,float lod){
  vec2 st=s/vec2(uW,uH)+.5;
  if(uIsFeed>.5){
    st.x=(st.x-.5)*(16./9.)/(4./3.)+.5;
    if(st.x<0.||st.x>1.) return vec3(.0);
  }
  return srgb2lin(textureLod(uScreen,st,lod).rgb)*uBright;
}

float windowGlow(vec2 q){
  vec2 c=.5*(WINDOW.xy+WINDOW.zw),h=.5*(WINDOW.zw-WINDOW.xy);
  vec2 d=abs(q-c);
  if(d.x>h.x||d.y>h.y) return 0.;
  float bars=smoothstep(.012,.02,abs(q.x-c.x))*smoothstep(.012,.02,abs(q.y-c.y-.12));
  return bars*(.75+.25*(q.y-WINDOW.y)/(2.*h.y));
}

// ---- materials: colour, roughness, specular, all computed from where the point is
struct Mat{ vec3 alb; float rough; float F0; int K; };

float lineMask(float x,float w){ return 1.-smoothstep(w*.5,w,abs(x)); }

// A board of sawn timber. The rings really are the distance to the tree's axis, cut by the plane of the board.
void wood(vec2 q,float px,float seedK,out vec3 col,out float gloss){
  float W=.187;
  float id=floor(q.y/W);
  float s0=hash1(id+seedK);
  float u=q.x+s0*7.3;
  float v=(fract(q.y/W)-.5)*W;
  float h=.022+.055*hash1(id+1.7+seedK);
  float cx=(hash1(id+2.3+seedK)-.5)*.11;
  float tilt=(hash1(id+3.1+seedK)-.5)*.055;
  float wob=.014*(vnoise(vec2(u*.9,id*3.1))-.5)+.0042*(vnoise(vec2(u*4.3,v*26.+id))-.5);
  float r=sqrt(pow(v-cx+wob,2.)+pow(h+u*tilt,2.));
  float rw=.0029+.0013*hash1(id+4.9+seedK);
  float f=r/rw+.5*vnoise(vec2(u*1.7,v*38.));
  float y=fract(f);
  // a year: pale fast spring wood darkening into the hard edge of autumn
  float late=smoothstep(.20,.94,y)*(1.-smoothstep(.94,1.,y));
  float yearId=floor(f);
  float tone=.82+.36*hash1(yearId+id*7.);
  // fibres
  float fib=vnoise(vec2(u*38.,v*1500.));
  float fib2=vnoise(vec2(u*11.,v*420.+3.));
  // open pores: long thin dashes, crowded in the spring wood
  vec2 pc=vec2(u*62.+hash1(floor(v*1400.))*9.,v*1400.);
  vec2 ci=floor(pc);
  float pr=hash21(ci);
  float dens=mix(.40,.06,smoothstep(.1,.5,y));
  vec2 pf=fract(pc)-.5;
  float pore=step(pr,dens)*(1.-smoothstep(.20,.46,abs(pf.x)))*(1.-smoothstep(.12,.42,abs(pf.y)));
  float pfade=1.-smoothstep(.0005,.0022,px);
  pore*=pfade;
  // medullary rays: faint flecks across the grain
  vec2 rc=vec2(u*210.,v*60.);
  float ray=smoothstep(.66,.95,vnoise(rc))*(1.-smoothstep(.0004,.0014,px))*.5;
  vec3 early=vec3(.47,.315,.165),lat=vec3(.235,.13,.062);
  col=mix(early,lat,late*.78*tone);
  col*=.84+.30*fib2;
  col*=.94+.12*fib;
  col=mix(col,col*.40,pore);
  col+=ray*.035*vec3(1.,.8,.55);
  col*=.86+.28*s0;
  col=mix(col,vec3(dot(col,vec3(.3,.6,.1))),.05);
  gloss=1.-.5*pore;
}

// dust, wear and handling on anything shiny. returns extra roughness, and how much dust sits at this point
vec2 handling(vec2 q,float scale){
  float sm=fbm(q*scale*7.,4);
  float prints=smoothstep(.46,.72,sm)*(.55+.45*sin(length(q*scale-vec2(.3,.2))*1400.+6.*vnoise(q*scale*40.)));
  vec2 cell=q*scale*300.;
  vec2 ci=floor(cell);
  float h=hash21(ci);
  // settled dust: mostly nothing, some motes, the odd bit of lint, thicker where nobody wipes
  float where=smoothstep(.40,.62,fbm(q*scale*3.3+9.,3));
  float speck=step(h,.0028*where+.0004)*(1.-smoothstep(.04,.30,length(fract(cell)-hash22(ci)*.6-.2)));
  return vec2(prints,speck);
}

Mat matOf(int m,vec3 p,vec3 n,float px){
  Mat M=Mat(bounceAlbedo(m),.8,.04,0);
  if(m==M_DESK){
    vec3 c; float gl;
    wood(vec2(p.x,p.z+.13),px,0.,c,gl);
    vec2 hd=handling(p.xz,1.);
    float wear=smoothstep(.09,.0,length((p.xz-vec2(-.12,.17))*vec2(.45,1.))-.05)*.5;
    M.alb=c;
    M.alb=mix(M.alb,vec3(.44,.42,.38),hd.y*.55);
    M.rough=clamp(.21+.05*fbm(p.xz*vec2(3.,9.),3)+.16*hd.x+.17*wear+(1.-gl)*.25+hd.y*.5,.05,1.);
    M.F0=.045;
    M.K=5;
  } else if(m==M_WALL){
    // an old coat of matt paint, rolled on by someone in a hurry
    float roll=fbm(p.xy*vec2(5.,2.1),3);
    float stip=vnoise(p.xy*850.);
    M.alb=vec3(.60,.58,.54)*(.90+.20*roll)*(.985+.03*stip);
    float scuff=smoothstep(.62,.8,fbm(p.xy*vec2(1.6,17.)+4.,3))*smoothstep(.1,-.2,p.y);
    M.alb*=1.-.12*scuff;
    M.rough=.82; M.F0=.03;
  } else if(m==M_BEZEL){
    // textured black plastic
    float g=vnoise(p.xy*5400.)*.6+vnoise(p.xy*2700.+9.)*.4;
    vec2 hd=handling(p.xy+vec2(.31,.17),1.8);
    M.alb=vec3(.0155)*(.92+.16*g);
    float dl=max(abs(p.x)-HW,abs(p.y)-HH);
    M.alb=mix(M.alb,vec3(.15,.145,.14),hd.y*smoothstep(.0028,.0045,dl));
    M.rough=.40+.10*g-.08*hd.x;
    M.F0=.042;
  } else if(m==M_STAND){
    // bead-blasted dark aluminium
    float g=vnoise(p.xz*vec2(5200.)+p.y*3000.);
    vec2 hd=handling(p.xz+p.y,1.2);
    M.alb=vec3(.018);
    M.alb=mix(M.alb,vec3(.2),hd.y*.7);
    M.rough=.38+.06*g+.10*hd.x; M.F0=.20;
    // only the foot, which looks up at the screen, needs to see it
    M.K=n.y>.7?3:0;
  } else if(m==M_MACH){
    float g=vnoise(p.xz*5200.);
    M.alb=vec3(.032,.032,.034)*(.94+.1*g);
    M.rough=.44+.05*g; M.F0=.12;
  } else if(m==M_KEY){
    vec3 q=kbLocal(p); q.y-=-q.z*.07;
    vec3 kc=kbCell(q);
    float cz=(kc.z+.5)*KU-.0572;
    vec2 k=vec2(q.x-kc.x,q.z-cz);
    M.alb=mix(vec3(.040,.040,.043),vec3(.030),step(.5,hash1(kc.x*91.+kc.z)));
    float wornK=smoothstep(.03,.0,length(q.xz-vec2(-.06,.0)))*.7+hash1(kc.x*17.+kc.z*3.)*.25;
    // printed legends, only marks at this distance
    float le=step(abs(k.x+kc.y*.45-.002),.0022)*step(abs(k.y+.0035),.003);
    M.alb+=vec3(.42)*le*(1.-wornK*.55);
    M.rough=mix(.48,.27,wornK); M.F0=.045; M.K=1;
  } else if(m==M_KCASE){
    float g=vnoise(p.xz*4200.);
    M.alb=vec3(.030)*(.85+.3*g); M.rough=.42; M.F0=.05; M.K=1;
  } else if(m==M_MUG){
    vec3 q=p-vec3(.17,DY,.31);
    float glaze=vnoise(q.xy*90.);
    M.alb=vec3(.085,.09,.10)*(.88+.24*glaze);
    float rim=smoothstep(.0845,.0872,q.y);
    M.alb=mix(M.alb,vec3(.20,.17,.14),rim);       // the glaze thins out at the lip
    M.rough=mix(.09,.3,rim); M.F0=.055; M.K=3;
  } else if(m==M_COFFEE){
    M.alb=vec3(.016,.009,.005); M.rough=.04; M.F0=.03; M.K=1;
  } else if(m==M_PAPER){
    vec3 q=p-vec3(-.262,DY,.042);
    vec2 b=vec2(.9877*(q.x-.012)+.1564*(q.z+.012),-.1564*(q.x-.012)+.9877*(q.z+.012));
    float top=q.y-.0070;
    float pad=step(0.,top)*step(abs(b.x),.074)*step(abs(b.y),.102);
    float rule=lineMask(fract(b.y/.0078)-.5,.04);
    vec2 a=vec2(.9744*q.x-.2250*q.z,.2250*q.x+.9744*q.z);
    float y2=(a.y+.148)/.0062;
    float lineI=floor(y2);
    float xl=a.x+.105;
    float len=.03+.165*hash1(lineI*.77);
    float txt=step(.26,fract(y2))*step(fract(y2),.74)*step(.018,xl)*step(xl,len)*step(.018,a.y+.148)
              *smoothstep(.25,.55,vnoise(vec2(xl*520.,lineI)));
    txt*=step(mod(lineI,9.),7.);
    vec3 sheet=vec3(.80,.79,.74)*(1.-.62*txt);
    // a few lines of pencil on the pad
    float yp=(b.y+.09)/.0078;
    float li=floor(yp);
    float xp=b.x+.06;
    float pen=step(li,9.)*step(0.,yp)*smoothstep(.25,.0,abs(fract(yp)-.3))*step(0.,xp)*step(xp,.03+.09*hash1(li*1.3))
              *smoothstep(.3,.6,vnoise(vec2(xp*340.,li*7.)));
    vec3 padc=vec3(.81,.80,.70)*(1.-.09*rule)*(1.-.5*pen);
    M.alb=mix(sheet,padc,pad);
    M.rough=.75; M.F0=.035;
  } else if(m==M_PEN){
    M.alb=vec3(.024); M.rough=.18; M.F0=.05; M.K=1;
  } else if(m==M_SPK){
    float g=vnoise(p.xy*2100.);
    M.alb=vec3(.042)*(.75+.5*g); M.rough=.5; M.F0=.04; M.K=1;
  } else if(m==M_CONE){
    vec2 f=p.xy-vec2(-.402,DY+.105-.032);
    float rib=.86+.14*sin(length(f)*1100.);
    M.alb=vec3(.026)*rib; M.rough=.55; M.F0=.04;
  } else if(m==M_CURT){
    float w=vnoise(p.xy*vec2(1400.,1500.));
    M.alb=vec3(.30,.31,.32)*(.9+.2*w);
    M.rough=.9; M.F0=.025;
  } else if(m==M_SHELF){
    if(abs(p.x+.07)>.395&&abs(p.x+.07)<.405&&p.y<.2545){ M.alb=vec3(.02); M.rough=.55; M.F0=.035; }
    else { M.alb=vec3(.52,.50,.45)*(.94+.1*fbm(p.xz*vec2(6.,14.),3)); M.rough=.5; M.F0=.04; }
  } else if(m==M_BOOK){
    float id=floor((p.x+.56)/.0345);
    float h=hash1(id+7.);
    // what books are bound in: mostly paper and cloth that have faded
    vec3 a=h<.28?vec3(.50,.47,.40):h<.46?vec3(.075,.075,.08):h<.62?vec3(.085,.115,.17):h<.74?vec3(.19,.07,.06):h<.90?vec3(.27,.23,.15):vec3(.10,.15,.125);
    a*=.78+.42*hash1(id+2.);
    // a paler band across some spines, where a title would go
    float ty=p.y-.30-.05*hash1(id+13.);
    if(hash1(id+19.)>.45) a=mix(a,vec3(.42,.40,.35),.55*smoothstep(.016,.013,abs(ty)));
    M.alb=a;
    M.rough=.6;
  } else if(m==M_CABLE){
    M.alb=vec3(.012); M.rough=.33; M.F0=.05; M.K=1;
  } else if(m==M_PLATE){
    M.alb=vec3(.44,.43,.40); M.rough=.34; M.F0=.04;
    vec2 q=p.xy-vec2(-.468,.058);
    float sockets=smoothstep(.0205,.0195,length(q));
    M.alb*=1.-.28*sockets;
  }
  return M;
}

// ---- gloss
vec3 vndf(vec3 Ve,float a,vec2 u){
  vec3 Vh=normalize(vec3(a*Ve.x,a*Ve.y,Ve.z));
  float l2=Vh.x*Vh.x+Vh.y*Vh.y;
  vec3 T1=l2>0.?vec3(-Vh.y,Vh.x,0.)/sqrt(l2):vec3(1.,0.,0.);
  vec3 T2=cross(Vh,T1);
  float r=sqrt(u.x),ph=6.2831853*u.y;
  float t1=r*cos(ph),t2=r*sin(ph);
  float s=.5*(1.+Vh.z);
  t2=(1.-s)*sqrt(max(1.-t1*t1,0.))+s*t2;
  vec3 Nh=t1*T1+t2*T2+sqrt(max(0.,1.-t1*t1-t2*t2))*Vh;
  return normalize(vec3(a*Nh.x,a*Nh.y,max(0.,Nh.z)));
}

// what a reflected ray sees
vec3 glimpse(vec3 p,vec3 L,float rough,int steps,int K){
  float lim=1e3; vec2 s=vec2(0.); bool toGlass=false; float tS=0.;
  if(L.z<-1e-3&&p.z>.0006){
    tS=-p.z/L.z; s=p.xy+L.xy*tS;
    if(abs(s.x)<HW&&abs(s.y)<HH){ toGlass=true; lim=tS-.001; }
  }
  float t; int m; vec3 n;
  if(traceRoom(p,L,lim,steps,t,m,n)){
    vec3 q=p+L*t;
    Mat M2=Mat(bounceAlbedo(m),.8,.04,0);
    vec3 c=M2.alb*lightAt(q,n,groupOf(m));
    if(m==M_WALL&&n.z<-.5) c+=gCW*windowGlow(q.xy);
    return c;
  }
  if(toGlass){
    float W=float(textureSize(uScreen,0).x);
    float fp=(rough*rough*2.+.002)*tS/uW*W/sqrt(float(K));
    return screenAt(s,max(0.,log2(max(fp,1.))));
  }
  return vec3(0.);
}

vec3 shade(vec3 ro,vec3 rd,float t,int m,vec3 n,vec2 jit){
  vec3 p=ro+rd*t;
  // how large this pixel is on the surface: a grazing look stretches it
  float px=t*uTan*2./uRes.y/max(abs(dot(rd,n)),.08);
  Mat M=matOf(m,p,n,px);
  int g=groupOf(m);
  vec3 E=lightAt(p,n,g);
  if(g==0) E*=ROOM_SPILL;
  float dlip=m==M_BEZEL?max(abs(p.x)-HW,abs(p.y)-HH):1.;
  if(dlip<.004) E*=mix(.05,1.,smoothstep(.0014,.004,dlip));   // the lip sees what's on screen, not the room's light
  vec3 c=M.alb*E;
  vec3 V=-rd;
  float NoV=max(dot(n,V),.02);
  if(M.K>0){
    vec3 T=normalize(cross(abs(n.y)<.9?vec3(0.,1.,0.):vec3(1.,0.,0.),n));
    vec3 B=cross(n,T);
    vec3 Ve=vec3(dot(V,T),dot(V,B),dot(V,n));
    float a=M.rough*M.rough;
    vec3 s=vec3(0.);
    vec3 po=p+n*.0012;
    int K=M.K;
    for(int i=0;i<K;i++){
      vec2 u=fract(jit+vec2(float(i)*.7548776662,float(i)*.5698402910));
      vec3 h=vndf(Ve,a,u);
      vec3 H=normalize(T*h.x+B*h.y+n*h.z);
      vec3 L=reflect(rd,H);
      float NoL=dot(n,L);
      if(NoL<=0.) continue;
      float VoH=max(dot(V,H),0.);
      float F=M.F0+(1.-M.F0)*pow(1.-VoH,5.);
      // Smith, for the way out
      float gl=2.*NoL/(NoL+sqrt(a*a+(1.-a*a)*NoL*NoL));
      s+=F*gl*glimpse(po,L,M.rough,m==M_DESK?26:22,K);
    }
    c+=s/float(K);
    // the part of the lobe a handful of rays can't reach
    c+=E*M.F0*.35*M.rough;
  } else {
    float Fv=M.F0+(1.-M.F0)*pow(1.-NoV,5.)*(1.-M.rough);
    c+=E*Fv;
    // the night window, large and far, as a soft sheen on whatever faces it
    vec3 Lm=reflect(rd,n);
    if(Lm.z>.05){
      vec2 q=p.xy+Lm.xy*(RMAX.z-p.z)/Lm.z;
      float blur=.25+M.rough*2.4;
      vec2 wc=.5*(WINDOW.xy+WINDOW.zw),wh=.5*(WINDOW.zw-WINDOW.xy);
      vec2 dd=abs(q-wc)-wh;
      float sheen=(1.-smoothstep(-blur,blur,dd.x))*(1.-smoothstep(-blur,blur,dd.y));
      c+=gCW*sheen*Fv;
    }
  }
  // the lip around the panel, lit edge-on by whatever the screen is showing there
  if(m==M_BEZEL){
    vec2 ad=abs(p.xy)-vec2(HW,HH);
    float dl=max(ad.x,ad.y);
    if(dl<.0028&&p.z>.0&&p.z<.0026){
      vec2 sn=clamp(p.xy,-vec2(HW,HH),vec2(HW,HH))*.995;
      vec3 sc=screenAt(sn,5.);
      vec3 base=(gC[0]+gC[1]+gC[2]+gC[3])*.25;
      float k=smoothstep(.0028,.0006,dl);
      c+=k*(.05*sc+.0011*base)*(.7+.6*vnoise(p.xy*vec2(70.,70.)));
    }
  }
  // the machine's light falls on its own front and on the desk
  if(uLed>0.){
    vec3 lp=vec3(LED_P,-.0525);
    vec3 w=lp-p; float d2=dot(w,w);
    if(d2<.09){
      float cs=max(dot(n,normalize(w)),0.);
      c+=M.alb*vec3(1.,.97,.92)*uLed*2.2e-5/(d2+1e-4)*(m==M_MACH?.25+cs:cs);
    }
  }
  return c;
}

void main(){
  vec2 v=(uv*2.-1.)*vec2(uAspect,1.);
  vec3 d=normalize(uF+v.x*uTan*uR+v.y*uTan*uU);
  // ---- the frame's light: what the screen shows now, as four corners of a rectangle
  for(int j=0;j<2;j++) for(int i=0;i<2;i++) gC[j*2+i]=texelFetch(uDrive,ivec2(i,j),0).rgb*uGlow;
  gCW=WIN_TINT*uGlow;
  // ---- monitor glass
  float t=-uO.z/d.z;
  vec3 h=uO+d*t;
  float sb=boxd(h.xy,vec2(uW,uH)*.5);
  float cover=smoothstep(.0006,-.0006,sb);
  vec3 glass=vec3(0.);
  if(cover>0.){
    vec2 s=h.xy/vec2(uW,uH)+.5;
    vec3 e;
    if(uIsFeed>.5){
      // HDMI: 4:3 panel pillarboxed on the 16:9 monitor
      vec2 p=vec2((s.x-.5)*(16./9.)/(4./3.)+.5,s.y);
      e=(p.x>0.&&p.x<1.)?texture(uScreen,p).rgb:vec3(0.);
    } else {
      e=texture(uScreen,s).rgb;
    }
    vec3 L=srgb2lin(e)*uBright;
    // the backlight is never quite shut, nor even: a little cloud in it, and glow toward the corners, more when seen off-axis
    vec2 c0=h.xy/vec2(HW,HH);
    float mura=fbm(h.xy*vec2(5.5,7.)+3.,3)-.5;
    float off=clamp(length(h.xy-uO.xy)/max(uO.z,.1),0.,1.);
    float corner=pow(max(abs(c0.x),abs(c0.y)),5.)*.5+pow(abs(c0.x*c0.y),3.)*1.3;
    L+=uBright*vec3(.95,.97,1.1)*(.00052+.0015*mura+corner*off*off*.0058);
    L*=1.+.42*mura;
    // real pixel grid sampled by a sensor without a low-pass filter
    float pitch=uW/(uRes.x*(uW/(2.*length(uO-h)*uTan*uAspect))*1.028);   // about one sensor pixel
    float nx=h.x/pitch, ny=h.y/pitch;
    vec3 m;
    m.r=.5+.5*cos(6.28318*(nx+.00));
    m.g=.5+.5*cos(6.28318*(nx+.333));
    m.b=.5+.5*cos(6.28318*(nx+.667));
    float rowm=.5+.5*cos(6.28318*ny*1.003);
    vec3 grid=m*(.62+.38*rowm)*1.7;
    L*=mix(vec3(1.),grid,.52*uMoire*uGrid);
    // dust on the glass, lit by what shines through it
    vec2 dc=h.xy*420.;
    vec2 di=floor(dc);
    float dh=hash21(di+17.);
    float dw=smoothstep(.42,.66,fbm(h.xy*6.+21.,3));
    float mote=step(dh,.0016*dw+.0003)*(1.-smoothstep(.03,.22,length(fract(dc)-.2-hash22(di)*.6)));
    L+=mote*uBright*(.0052+.05*srgb2lin(textureLod(uScreen,clamp(s,0.,1.),4.).rgb));
    glass=L;
  }
  float aRoom=0.; vec3 room=vec3(0.);
  // anything standing between the camera and the lowest part of the glass?
  bool front=cover>0.&&min(h.y,uO.y)<-.125;
  if(cover<1.||front){
    vec2 px=vec2(2.*uAspect/uRes.x,2./uRes.y);
    float hit=0.;
    const vec2 SUB[4]=vec2[4](vec2(-.125,-.375),vec2(.375,-.125),vec2(-.375,.125),vec2(.125,.375));
    float idSum=0.; float idN=0.;
    int ns=cover<1.?4:1;
    vec2 js=hash22(gl_FragCoord.xy+vec2(uT*61.,uT*37.));
    for(int i=0;i<ns;i++){
      vec2 w=v+SUB[i]*px;
      vec3 dd=normalize(uF+w.x*uTan*uR+w.y*uTan*uU);
      float tt; int mm; vec3 nn;
      float lim=1e3;
      if(cover>=1.) lim=-uO.z/dd.z-.003;
      if(traceRoom(uO,dd,lim,cover<1.?110:40,tt,mm,nn)){
        vec3 q=uO+dd*tt;
        // the front of the panel, seen through its own edge: that's the glass, already drawn
        if(mm==M_BEZEL&&q.z<.00035&&abs(q.x)<=HW+.0004&&abs(q.y)<=HH+.0004){
          room+=vec3(.0002); aRoom+=1.;
          idSum+=1./max(dot(q-uO,uF),.01); idN+=1.;
          continue;
        }
        room+=shade(uO,dd,tt,mm,nn,fract(js+vec2(i)*.37));
        aRoom+=1.;
        idSum+=1./max(dot(q-uO,uF),.01); idN+=1.;
      }
    }
    if(idN>0.){
      room/=float(ns);
      if(cover>=1.) cover=cover*(1.-aRoom/float(ns));
      oRoom=vec4(room,idSum/idN);
    } else oRoom=vec4(0.);
  } else {
    oRoom=vec4(0.);
  }
  o=vec4(glass*cover,cover);
}
