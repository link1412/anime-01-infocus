#version 410
// Density + paper -> display-referred colour. Magnification resolves the ink into characters:
// each cell of a jittered grid holds a glyph whose local mean equals the painting's ink, so the mip
// chain makes the two the same thing when zoomed out. The sea comes in here too, as ink.
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uD,uReg,uPaper,uNoise,uMist,uFrom;
uniform sampler2DArray uAtlas;
uniform float uMean[11];
uniform float uAspect,uZoom,uPhPaper,uPaperP,uCell,uPhGlyph,uAtlasRes,uSettle;
uniform float uHasFrom,uReveal,uHorizon,uSunR;
uniform vec2 uZc,uSunC;
uniform vec3 uKRed;     // vermilion extinction; during the arrival it passes through gold and orange
const vec3 kInk=vec3(.98,1.,1.03)*1.15, kInd=vec3(1.30,1.02,.74);

vec4 paperAt(vec2 X){
  vec2 q=X/uPaperP;
  vec4 a=texture(uPaper,q,.35);
  vec4 b=texture(uPaper,q*7.13+.37,.35);
  vec4 c=texture(uPaper,q*43.7+.11,.35);
  return vec4(a.r+(b.r-.5)*.8+(c.r-.5)*.6, a.g+(b.g-.5)*.6+(c.g-.5)*.4, a.b, clamp(a.a+b.a*.7+c.a*.5,0.,1.));
}
vec3 odOf(vec3 d,float gran){
  return max(d.r,0.)*gran*kInk+max(d.b,0.)*gran*kInd+max(d.g,0.)*(1.+(gran-1.)*.45)*uKRed;
}
float glyph(float layer,vec2 auv,float lod){ return textureLod(uAtlas,vec3(auv,layer),lod).r; }

void main(){
  vec2 p=vec2((uv.x*2.-1.)*uAspect,uv.y*2.-1.);
  vec2 v=uZc*(1.-1./uZoom)+p/uZoom;
  vec2 X=vec2(v.x+uPhPaper,v.y);
  vec4 pa=paperAt(X);
  float cloud=texture(uPaper,X/(uPaperP*3.7)+.2,3.).r;
  // ---- warm aged xuan
  vec3 base=vec3(.905,.852,.742);
  float macro=smoothstep(6.,40.,uZoom);
  float tone=(pa.r-.5)*mix(.052,.11,macro)+(cloud-.5)*.036+(pa.g-.5)*mix(.035,.13,macro);
  vec3 paper=base*(1.+tone*vec3(.9,1.,1.25));
  paper=mix(paper,base*vec3(1.05,1.05,1.06),pa.a*.26);
  float edge=smoothstep(.76,1.02,abs(v.y));
  paper*=1.-edge*.030*vec3(.8,1.,1.5);

  // ---- ink: fibres resist it a little, pools fill the tooth
  vec3 d=texture(uD,uv).rgb;
  float gran=1.+(pa.b-.5)*.22-pa.a*.20;
  vec3 od=odOf(d,gran);
  vec3 col=paper*exp(-od)+vec3(.012,.011,.012)*(1.-exp(-od));

  // ---- the painting was written, not painted
  float zw=smoothstep(2.4,10.,uZoom);
  if(zw>0.){
    vec2 Xg=vec2(v.x+uPhGlyph,v.y);
    vec2 g=Xg/uCell, id=floor(g), f=g-id;
    vec2 idm=mod(id,2048.);
    vec2 h=hash22(idm);
    float h3=hash21(idm+17.),h4=hash21(idm+41.),h5=hash21(idm+83.);
    vec2 jit=(h-.5)*.22;
    float ang=(h3-.5)*.46;
    float sc=.86+.20*h4;
    float var=floor(h5*2.999);
    // this cell's own ink, sampled at its centre, so that one character is one load of the brush
    vec2 cv=vec2((id.x+.5+jit.x)*uCell-uPhGlyph,(id.y+.5+jit.y)*uCell);
    vec2 cp=(cv-uZc*(1.-1./uZoom))*uZoom;
    vec2 cuv=vec2(cp.x/uAspect,cp.y)*.5+.5;
    vec3 dc=texture(uD,cuv).rgb;
    float rg=texture(uReg,cuv).a;
    vec3 odc=odOf(dc,1.);
    float zt=smoothstep(7.,26.,uZoom);
    vec3 odp=mix(od,odc,zt*.9);
    float odg=max(odp.g,1e-4);
    // solid ink of this character: same hue as the ink here, loaded dark
    float Ds=max(1.75,odg);
    vec3 solid=odp*(Ds/odg);
    float frC=(1.-exp(-odg))/(1.-exp(-Ds));          // this cell's load
    float frP=(1.-exp(-max(od.g,0.)))/(1.-exp(-Ds)); // the wash under it, continuous across cells
    // which character: 水 in the water, 云 in mist and on bare paper, 日 in the sun, 山 elsewhere
    float wWater=smoothstep(-.35,-.8,rg);
    float wCloud=max(smoothstep(.22,.5,rg),smoothstep(.10,.03,odc.g))*(1.-wWater);
    float red=max(dc.g,0.)/(max(dc.r+dc.g+dc.b,1e-3));
    float wSun=smoothstep(.45,.8,red);
    vec2 q=f-.5-jit;
    float cs=cos(ang),sn=sin(ang);
    q=vec2(cs*q.x+sn*q.y,-sn*q.x+cs*q.y)/sc;
    // edges roughen as the magnification finds the brush
    vec2 auv=q+.5+(vec2(texture(uNoise,X*17.3).g,texture(uNoise,X*17.3+.37).g)-.5)*.022*zt;
    float cellPx=uCell*uZoom*(uAspect>0.?477.:477.)/1.45;
    float lod=max(log2(uAtlasRes/max(cellPx*sc,1.))+.15,0.);
    float vr=var;
    float C0=glyph(vr,auv,lod), M0=uMean[int(vr)];
    float C1=glyph(3.+vr,auv,lod), M1=uMean[3+int(vr)];
    float C2=glyph(6.+vr,auv,lod), M2=uMean[6+int(vr)];
    float C3=glyph(9.+min(vr,1.),auv,lod), M3=uMean[9+int(min(vr,1.))];
    float C=mix(mix(mix(C0,C2,wCloud),C1,wWater),C3,wSun);
    float mu=mix(mix(mix(M0,M2,wCloud),M1,wWater),M3,wSun)*sc*sc;
    // dark ink makes bold characters: the same glyph, thickened through its blurred mip
    float vis=smoothstep(2.6,9.,uZoom);
    float lb=lod+1.7;
    float B=mix(mix(mix(glyph(vr,auv,lb),glyph(6.+vr,auv,lb),wCloud),glyph(3.+vr,auv,lb),wWater),glyph(9.+min(vr,1.),auv,lb),wSun);
    float bold=smoothstep(.18,.85,frC)*vis;
    float Cb=max(C,smoothstep(.10,.42,B));
    // at low magnification the glyph is below resolution: it must equal its own mean
    float muB=mix(mu,mu*2.1,bold);
    C=mix(muB,mix(C,Cb,bold),vis);
    C*=mix(1.,.82+.36*pa.b,zt);
    // wet ink on xuan: a feathered halo creeps out of each stroke along the fibres
    float capil=smoothstep(.06,.5,B+(pa.a-.35)*.55+(texture(uNoise,X*31.).g-.5)*.35)*.24*zt;
    // ghost writing where there is hardly any ink at all
    frC=max(frC,.055*smoothstep(8.,24.,uZoom));
    // magnified, the wash between the characters thins: the writing is what remains
    float fill=mix(1.,.10,smoothstep(4.,18.,uZoom));
    float mean=min(frC/max(muB,1e-3),1.);                 // mean-preserving load when small
    float direct=clamp(pow(frC,.72)*1.25,0.,1.);          // one load of the brush per character when large
    float gc=clamp(max(C,capil)*mix(mean,direct,smoothstep(5.,16.,uZoom)),0.,1.);
    float fc=clamp(max(frP-muB,0.)/max(1.-muB,1e-3)*fill,0.,1.);
    float cov=gc+(1.-gc)*fc;
    vec3 wr=paper*mix(vec3(1.),exp(-solid),cov);
    col=mix(col,wr,zw);
  }

  // ---- the sea, translated. Colour drains, tone becomes ink, ink thresholds away.
  if(uHasFrom>.5&&uReveal<.9995){
    vec3 L=texture(uFrom,uv).rgb;
    vec3 src=pow(1.-exp(-L*1.1),vec3(1./2.2));
    float g=luma(src);
    float r=uReveal;
    // the water front runs down the sheet and is feathered by the fibres
    float big=texture(uMist,vec2(v.x/7.3+.11,v.y/3.7+.4)).g;
    float fine=texture(uMist,vec2(v.x/.9+.3,v.y/.7)).r;
    float fib=texture(uPaper,X/uPaperP,1.).a;
    float order=clamp((1.-uv.y)*.56+big*.22+fine*.12+fib*.10,0.,1.);
    float dS=length(v-uSunC);
    float disc=1.-smoothstep(uSunR*.96,uSunR*1.20,dS);
    float W1=smoothstep(0.,.52,r*2.7-order)*(1.-disc*.95);            // colour drains
    float W2=smoothstep(0.,.44,r*2.25-.30-order*.85);                 // tone becomes ink on paper
    vec3 drained=mix(src,vec3(g)*vec3(1.04,1.,.92),W1);
    // tone as ink density relative to paper
    float Dsrc=max(-log(max(g,.012)/.86),0.);
    float s1=smoothstep(.25,.90,r);
    float tau=1.5*s1+4.*pow(s1,6.);
    // waves turn into lines: as the wash thins, the dark that is left gathers along brush tracks
    float track=texture(uNoise,vec2(X.x*.045,v.y*.62)).a*.7+texture(uNoise,vec2(X.x*.022+.3,v.y*1.3)).a*.3;
    float tight=smoothstep(.28,.72,r);
    float lineM=mix(1.,smoothstep(.70,.86,track),tight);
    float seaOrSky=mix(1.,lineM,step(v.y,uHorizon));
    float D2=max(Dsrc-tau*.55,0.)*mix(1.,1.5,smoothstep(.2,.7,r))*seaOrSky*smoothstep(1.,.74,r);
    vec3 wash=paper*exp(-D2*kInk*.8);
    // paper fibre rises through the picture before the picture has gone
    float rise=smoothstep(.06,.60,r);
    vec3 fibred=drained*mix(vec3(1.),(1.+tone*vec3(.9,1.,1.25)*1.7)*mix(1.,1.07,pa.a*.5)*(1.+(pa.b-.5)*.05),rise);
    vec3 c0=mix(fibred,wash,W2);
    // the world's ink lands on top of whatever is left of the sea
    vec3 odr=od*smoothstep(0.,.07,r);
    col=c0*exp(-odr)+vec3(.012,.011,.012)*(1.-exp(-odr));
  }
  // ---- the paper's tones settle on three values, the middle one most of all, so that the picture's own histogram
  // is the character for mountain. The shift is under half of one display code value: only the histogram can see it.
  {
    const float HMU=.8535,HLAM=.0125,HA=.0016;
    float l=luma(col);
    float w=smoothstep(HMU-2.2*HLAM,HMU-1.6*HLAM,l)*(1.-smoothstep(HMU+1.6*HLAM,HMU+2.2*HLAM,l));
    col*=1.-uSettle*HA*w*sin(6.28318*(l-HMU)/HLAM)/max(l,.1);
  }
  o=vec4(col,1.);
}
